import os
import numpy as np
import gym
import safety_gym
# import bullet_safety_gym
# import safety_gym
import torch
from torch.utils.tensorboard import SummaryWriter

from dmbrl.agent import PPOAgent
from dmbrl.trainer import onpolicy_trainer
from dmbrl.data import Collector, Storage
from dmbrl.env import VectorEnv, SubprocVectorEnv
from dmbrl.network import ActorProb2, Critic
from dmbrl.utils import Config, set_seed, BaseNormalizer, MeanStdNormalizer, tensor

def get_args(parser):
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--hyperparams', type=str, default='hyper_params/PPO_CarRacing.yaml')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--cudaid', type=int, default=-1)
    parser.add_argument("--mixed", default=True, action="store_false")
    parser.add_argument("--contentbased", default=False, action="store_true")


from CarRacing_wrapper import CarRacingWrapper, RacingNet
from Munit.load_munit import load_munit_model

def main():
    cfg = Config()
    get_args(cfg.parser)
    cfg.load_hyperparams()
    cfg.select_device()

    HP = cfg.hyperparams

    content_extractor_fn = None
    if HP['misc']['contentbased']:
        munit = load_munit_model('Pong', Config.DEVICE)
        content_extractor_fn = munit.infer

    env = CarRacingWrapper(gym.make(HP['env']['name']))
    # env.spec.reward_threshold = 10.0
    HP['network']['s_dim'] = env.observation_space.shape
    HP['network']['a_dim'] = env.action_space.shape or env.action_space.n
    HP['agent']['action_range'] = [env.action_space.low, env.action_space.high]

    print('obs_space', HP['network']['s_dim'])
    print('act_space', HP['network']['a_dim'])

    # initialize train/test env
    if not HP['misc']['mixed']:
        train_envs = SubprocVectorEnv(
            [lambda: CarRacingWrapper(gym.make(HP['env']['name']), wait_len=30) for _ in range(HP['env']['num_env_train'])]
        )
        test_envs = SubprocVectorEnv(
            [lambda: CarRacingWrapper(gym.make(HP['env']['name'])) for _ in range(HP['env']['num_env_test'])]
        )
    else:
        train_envs = SubprocVectorEnv(
            [lambda: CarRacingWrapper(gym.make(HP['env']['name']), wait_len=30) for _ in range(HP['env']['num_env_train']//2)]
            + [lambda: CarRacingWrapper(
                gym.make(HP['env']['name']), wait_len=30, change_style=True, munit_model_fn=content_extractor_fn
                ) for _ in range(HP['env']['num_env_train']//2)
            ]
        )
        test_envs = SubprocVectorEnv(
            [lambda: CarRacingWrapper(gym.make(HP['env']['name'])) for _ in range(HP['env']['num_env_test']//2)]
            + [lambda: CarRacingWrapper(
                gym.make(HP['env']['name']), change_style=True, munit_model_fn=content_extractor_fn
            ) for _ in range(HP['env']['num_env_train']//2)]
        )

    # seed
    set_seed(HP['misc']['seed'])
    train_envs.seed(HP['misc']['seed'])
    test_envs.seed(HP['misc']['seed'])

    # model
    actor = RacingNet(HP['network']['s_dim'], HP['network']['a_dim']).to(Config.DEVICE)
    critic = None
    # critic = RacingCritic(HP['network']['s_dim']).to(Config.DEVICE)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=HP['agent']['actor_lr'])
    critic_optim = None
    # critic_optim = torch.optim.Adam(critic.parameters(), lr=HP['agent']['critic_lr'])
    
    agent = PPOAgent(
        actor, critic,
        actor_optim, critic_optim,
        torch.distributions.Normal, 
        HP['agent']['gamma'],
        gae_factor=HP['agent']['gae_lambda'], 
        max_grad_norm=HP['agent']['max_grad_norm'],
        eps_clip_ratio=HP['agent']['clip_epsilon'],
        vf_coef=HP['agent']['Value_fn_coef'], 
        ent_coef=HP['agent']['entropy_coef'],
        # action_range=HP['agent']['action_range'],
        state_normalizer=BaseNormalizer(),
        ignore_done=HP['agent']['ignore_done'],
    )
    
    # save model
    if not os.path.exists(f"{HP['trainer']['model_dir']}/{HP['env']['name']}"):
        os.mkdir(f"{HP['trainer']['model_dir']}/{HP['env']['name']}", 0o777)
    save_path = f"{HP['trainer']['model_dir']}/{HP['env']['name']}/ppo_mixed"


    # trainer
    def stop_fn(r):
        return False # r > env.spec.reward_threshold

    if not HP['misc']['test']:
        # collector
        train_collector = Collector(agent, train_envs, Storage(HP['trainer']['replay_size']), \
            store_keywords=None, act_space=env.action_space)
        test_collector = Collector(agent, test_envs)


        # log
        writer = SummaryWriter(f"{HP['trainer']['writer_dir']}/{HP['env']['name']}/ppo_mixed")

        ## onpolicy_trainer
        result = onpolicy_trainer(
            agent, 
            train_collector, 
            test_collector,
            HP['trainer']['warmup_episode'],
            HP['trainer']['epoch'],
            HP['trainer']['step_per_epoch'],
            HP['trainer']['collect_episode_per_step'],
            HP['trainer']['n_train_repeat'],
            HP['trainer']['batch_size'],
            HP['trainer']['test_episode'],
            save_path,
            stop_fn,
            writer,
        )

        train_collector.close()
        test_collector.close()

    else:
        # from carl.envs.box2d.carl_vehicle_racing import CARLVehicleRacingEnv

        env_base = CarRacingWrapper(gym.make(HP['env']['name']), wait_len=1000)
        env_new = CarRacingWrapper(gym.make(HP['env']['name']), wait_len=1000, change_style=True)
        # env_newstyle = CARLVehicleRacingEnv(contexts={0:{'VEHICLE':15}})
        env_base.seed(HP['misc']['seed'])
        env_new.seed(HP['misc']['seed'])
        # env_newstyle.seed(HP['misc']['seed'])

        agent.load_model(save_path)
        agent.eval()
        
        # n_episode = 1
        # save_paths = ['../data/base.npy', '../data/rainy.npy']
        # generate_dataset(env_base, agent, n_episode, save_paths)
        
        collector = Collector(agent, env_new, Storage(HP['trainer']['replay_size']))
        result = collector.collect(n_episode=10, render=HP['misc']['render']) # 
        reward_list = result["reward_list"]
        print(f'Final reward: {np.mean(reward_list)}±{np.std(reward_list)}, length: {result["length"]}')

        # replay = collector.replay
        # batch, _ = replay.sample(0)
        # cost = batch.cost
        # print(f'min:{cost.min()}, max:{cost.max()}, non-zeros:{cost[np.nonzero(cost)].shape}, mean:{cost.mean()}')


import imageio 
import torch.distributions as D
from CarRacing_wrapper import add_rain


def generate_dataset(env_base, agent, n_episode, save_paths):
    
    render_base, render_new = [], []

    for _ in range(n_episode):
        s = env_base.reset()
        _cnt = 0

        slant = np.random.randint(-5,5)

        while True:
            _, alpha, beta = agent.actor(np.array([s]))
            dist = D.Beta(alpha, beta)
            act = dist.sample().detach().cpu().numpy()[0]

            s, _, done, _ = env_base.step(act)

            frame_base = env_base.render("state_pixels")
            frame_new = add_rain(frame_base.copy(), slant)

            render_base.append(frame_base)
            render_new.append(frame_new)

            _cnt += 1
            # if _cnt > 60:
            #     break

            if done:
                break

    render_base = np.array(render_base)
    render_new = np.array(render_new)

    for path, dataset in zip(save_paths, [render_base, render_new]):
        np.save(path, dataset)

    imageio.mimsave('base.gif', render_base[:500,:,:], 'GIF', fps=30.0)
    # imageio.mimsave('new.gif', render_new, 'GIF', duration=0.01)



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()