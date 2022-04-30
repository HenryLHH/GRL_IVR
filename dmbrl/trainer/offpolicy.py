import time
import numpy as np
from tqdm import tqdm

from dmbrl.trainer.utils import test_episode, gather_info

def offpolicy_trainer(
    agent, train_collector, test_collector, max_epoch,
    step_per_epoch, collect_per_step, 
    episode_per_test, batch_size, save_path,
    stop_fn=None, writer=None, verbose=False,
    # train_fn=None, test_fn=None
):
    
    global_step = 0
    global_length = 0
    best_epoch, best_reward = -1, -1

    lr_cost_coef = 5e-3
    max_cost_coef = 20

    constraint_alpha = 25.0

    n_episode_warmup = 16

    print('------ Start warm-up. ------')
    train_collector.collect(n_episode=n_episode_warmup, random=True)

    start_time = time.time()

    for epoch in range(1, max_epoch + 1):
        
        postfix_data = {}
        # train
        agent.train()

        with tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}') as t:
            while t.n < t.total:                
                result = train_collector.collect(n_step=collect_per_step)

                for _ in range(min(result['n_step'] // collect_per_step,
                                t.total - t.n)):
                    global_step += 1
                    
                    batch_data, indices = train_collector.sample(batch_size, return_indices=True)
                    losses = agent.learn(batch_data)
                    agent.post_process_fn(batch_data, train_collector.replay, indices)

                if writer:
                    for k, _loss in losses.items():
                        if isinstance(_loss, tuple):
                            data_dict = dict(zip(['mu', 'mu-sigma', 'mu+sigma'], _loss))
                            writer.add_scalars(k, data_dict, global_step=global_step)
                        else:
                            writer.add_scalar(k, _loss, global_step=global_step)
                    
                    if result['length'] > 0:

                        # # update cost_coef
                        # cost_coef_para = agent.cost_coef_para
                        # grad = np.exp(cost_coef_para) / (1+np.exp(cost_coef_para))
                        # cost_coef_para += lr_cost_coef * grad * (result['cost'] - constraint_alpha)
                        # agent.cost_coef_para = cost_coef_para
                        # cost_coef = np.log(1+np.exp(cost_coef_para))
                        # cost_coef = np.clip(cost_coef, 0.0, max_cost_coef)

                        if 'reward_list' in result:
                            for r, c, l in zip(result['reward_list'], result['cost_list'], result['length_list']):
                                global_length += l
                                writer.add_scalar('reward', r, global_step=global_length)
                                writer.add_scalar('cost', c, global_step=global_length)

                        else:
                            global_length += result['length']
                            writer.add_scalar('reward', result['reward'], global_step=global_length)
                            writer.add_scalar('cost', result['cost'], global_step=global_length)
                            writer.add_scalar('cost_coef', cost_coef, global_step=global_length)
                        
                        postfix_data['cost'] = f'{result["cost"]:.2f}'
                        postfix_data['last_reward'] = f'{result["reward"]:.2f}'
                        postfix_data['lambda'] = f'{losses["cost_coef"]:.2f}'
                
                t.update(1)
                t.set_postfix(**postfix_data)

            if t.n < t.total:
                t.update()
        
        # save model
        agent.save_model(save_path)

        # test
        result = test_episode(agent, test_collector, episode_per_test)
        if best_epoch == -1 or best_reward < result['reward']:
            best_reward = result['reward']
            best_epoch = epoch
        
        if verbose:
            print(f'Epoch #{epoch}| test_reward: {result["reward"]:.3f}, '
                f'best_reward: {best_reward:.3f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector, best_reward)



def offpolicy_trainer2(
    agent, train_collector, test_collector, max_epoch,
    step_per_epoch, collect_per_step, 
    episode_per_test, batch_size, save_path,
    stop_fn=None, writer=None, verbose=False,
    # train_fn=None, test_fn=None
):
    
    global_step = 0
    global_length = 0
    best_epoch, best_reward = -1, -1

    lr_cost_coef = 5e-3
    max_cost_coef = 20

    constraint_alpha = 25.0

    n_episode_warmup = 16

    print('------ Start warm-up. ------')
    train_collector.collect(n_episode=n_episode_warmup, random=True)

    start_time = time.time()

    for epoch in range(1, max_epoch + 1):
        
        postfix_data = {}
        # train
        agent.train()

        with tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}') as t:
            while t.n < t.total:
                result = train_collector.collect(n_episode=8)

                if 'reward_list' in result:
                    for r, c, l in zip(result['reward_list'], result['cost_list'], result['length_list']):
                        global_length += l
                        writer.add_scalar('reward', r, global_step=global_length)
                        writer.add_scalar('cost', c, global_step=global_length)

                postfix_data['cost'] = f'{result["cost"]:.2f}'
                postfix_data['reward'] = f'{result["reward"]:.2f}'

                for _ in range(int(result['length']) // collect_per_step):
                    global_step += 1
                    
                    batch_data, indices = train_collector.sample(batch_size, return_indices=True)
                    losses = agent.learn(batch_data)
                    agent.post_process_fn(batch_data, train_collector.replay, indices)

                    if writer:
                        for k, _loss in losses.items():
                            if isinstance(_loss, tuple):
                                data_dict = dict(zip(['mu', 'mu-sigma', 'mu+sigma'], _loss))
                                writer.add_scalars(k, data_dict, global_step=global_step)
                            else:
                                writer.add_scalar(k, _loss, global_step=global_step)
                    
                    
                    postfix_data['lambda'] = f'{losses["cost_coef"]:.2f}'
                    
                    t.update(1)
                    t.set_postfix(**postfix_data)

            if t.n < t.total:
                t.update()
        
        # save model
        agent.save_model(save_path)

        # test
        result = test_episode(agent, test_collector, episode_per_test)
        if best_epoch == -1 or best_reward < result['reward']:
            best_reward = result['reward']
            best_epoch = epoch
        
        if verbose:
            print(f'Epoch #{epoch}| test_reward: {result["reward"]:.3f}, '
                f'best_reward: {best_reward:.3f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector, best_reward)
        