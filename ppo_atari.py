import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from atari_wrapper import make_atari_env, wrap_deepmind
from CarRacing_wrapper import make_carracing_env
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import ICMPolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch.nn as nn

from Munit.load_munit import load_munit_model

class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, np.prod(action_shape))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, output_dim),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=1000)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="exp/log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--logger", type=str, default="tensorboard")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--mixed", default=True, action="store_false")
    parser.add_argument("--contentbased", default=False, action="store_true")
    
    return parser.parse_args()


def test_ppo(args=get_args()):
    content_extractor_fn = None
    if args.contentbased:
        munit = load_munit_model('Pong', args.device)
        content_extractor_fn = munit.infer

    if args.task == 'PongNoFrameskip-v4':
        env, train_envs, test_envs = make_atari_env(
            args.task,
            args.seed,
            args.training_num,
            args.test_num,
            mixed=args.mixed,
            munit_model_fn=content_extractor_fn,
        )
    elif args.task == "CarRacing-v0":
        env, train_envs, test_envs = make_carracing_env(
            args.task,
            args.seed,
            args.training_num,
            args.test_num,
            mixed=args.mixed,
        )
    else:
        raise RuntimeError('No such envs.')

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    net = DQN(
        *args.state_shape,
        args.action_shape,
        device=args.device,
        features_only=True,
        output_dim=args.hidden_size
    )
    actor = Actor(net, args.action_shape, device=args.device, softmax_output=False)
    critic = Critic(net, device=args.device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    # define policy
    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    ).to(args.device)
    
    
    # load a previous policy
    # if args.resume_path:
    #     policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #     print("Loaded agent from: ", args.resume_path)
    

    # log
    algor_name = 'ppo' if not args.mixed else 'ppo_mixed'
    log_name = os.path.join(args.task, algor_name)
    log_path = os.path.join(args.logdir, log_name)


    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth'), map_location=args.device))
        print("Loaded agent.")
        
        policy.eval()
        print("Testing agent ...")
        
        test_collector = Collector(policy, test_envs, exploration_noise=True)
        test_collector.reset()
        result = test_collector.collect(n_episode=10, render=args.render)
        rew, std = result["rews"].mean(), result["rews"].std()
        print(f"Reward (over {result['n/ep']} episodes): {rew}±{std}")
        print(result["rews"])


    if args.watch:
        watch()
        exit(0)


    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif "Pong" in args.task:
            return mean_rewards >= 20
        else:
            return False

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    test_ppo(get_args())