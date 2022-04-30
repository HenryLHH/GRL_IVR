import time
import numpy as np
from tqdm import tqdm

from dmbrl.trainer.utils import test_episode, gather_info


def onpolicy_trainer(
    agent, 
    train_collector, 
    test_collector, 
    warmup_episode,
    max_epoch,
    step_per_epoch, 
    collect_per_step, 
    n_train_repeat,
    batch_size, 
    episode_per_test, 
    save_path,
    stop_fn=None, 
    writer=None, 
    verbose=True,
    # train_fn=None, test_fn=None
):

    global_optim_step = 0 # the times of optimization (for each batch)
    global_length = 0 # the times of env taking action to step
    best_epoch, best_reward = -1, -1

    print('------ Start warm-up. ------')
    train_collector.collect(n_episode=warmup_episode, random=True)
    agent.learn(train_collector.sample(0), batch_size, n_train_repeat)
    print('------ End warm-up. ------')

    start_time = time.time()

    for epoch in range(1, max_epoch + 1):
        
        postfix_data = {}
        # train
        agent.train()

        with tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}') as t:
            while t.n < t.total:
                result = train_collector.collect(n_episode=collect_per_step)
                   
                losses = agent.learn(train_collector.sample(0), batch_size, n_train_repeat)
                agent.sync_weights()
                train_collector.reset_replay()
                
                step = 1
                for k in losses.keys():
                    if isinstance(losses[k], list):
                        step = max(step, len(losses[k]))
                
                if writer:
                    # num of optimization steps
                    _optim_step = 1
                    for k in losses.keys():
                        if isinstance(losses[k], list):
                            _optim_step = max(len(losses[k]), _optim_step)

                    for k, _loss in losses.items():
                        if isinstance(_loss, list):
                            for _ in range(len(_loss)):
                                writer.add_scalar(k, _loss[_], global_step=global_optim_step+_+1)
                        elif isinstance(_loss, tuple):
                            for _, data in enumerate(zip(*_loss)):
                                data_dict = dict(zip(['mu', 'mu-sigma', 'mu+sigma'], data))
                                writer.add_scalars(k, data_dict, global_step=global_optim_step+_+1)
                        else:
                            writer.add_scalar(k, _loss, global_step=global_optim_step+_optim_step)
                    global_optim_step += _optim_step


                    if 'reward_list' in result:
                        for r, l in zip(result['reward_list'], result['length_list']):
                            global_length += l
                            writer.add_scalar('reward', r, global_step=global_length)
                    
                    else:
                        global_length += result['length']
                        writer.add_scalar('reward', result['reward'], global_step=global_length)
                
                postfix_data['reward'] = f'{result["reward"]:.2f}'
                
                t.update(step)
                t.set_postfix(**postfix_data)

            if t.n <= t.total:
                t.update()

        # max_cost_coef *= decay        

        # test
        result = test_episode(agent, test_collector, episode_per_test)
        if best_epoch == -1 or best_reward < result['reward']:
            best_reward = result['reward']
            best_epoch = epoch
            # save model
            agent.save_model(save_path)

        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["reward"]:.3f}, '
                  f'best_reward: {best_reward:.3f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector, best_reward)