import torch
import time
import numpy as np

from dmbrl.data import Batch, Storage, CacheBuffer
from dmbrl.env import BaseMultiEnv

class Collector:

    def __init__(self, agent, env, replay=None,
                    act_space=None, store_keywords=None):
        """Collector

        args:
            agent: agent of reinforcement learning
            env: environment to collect
            replay: experience pool
            store_keywords: variable to be stored after each step of agent
        
        return:
            None
        """
        self.agent = agent
        self.process_fn = agent.process_fn
        self.env = env
        self._multi_env = False
        self.env_num = 1
        if isinstance(env, BaseMultiEnv):
            self._multi_env = True
            self.env_num = len(env)
        self.replay = replay if replay is not None else Storage(1000)
        self.act_space = act_space
        self.store_keywords = store_keywords

        self.states = None
        self.has_reset_fn = hasattr(self.agent, 'reset')

        if self._multi_env:
            self._cached_replays = [CacheBuffer() for _ in range(self.env_num)]

        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0.0

        self.reset_env()
        self.reset_replay()
    

    def reset_env(self):
        self._obs = self.agent.state_normalizer(self.env.reset())
        self._act = self._rew = self._done = self._info = None

        # rew is one-step reward, reward is sum of rew
        if self._multi_env:
            self.cum_reward = np.zeros(self.env_num)
            self.length = np.zeros(self.env_num)
        else:
            self.cum_reward = self.length = 0

    def reset_replay(self):
        if self._multi_env:
            for r in self._cached_replays:
                r.reset()
        self.replay.reset()


    def _reset_states(self, index=None):
        if hasattr(self.agent, 'reset_states'):
            self.agent.reset_states(self.states, index)


    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


    def collect(self, n_step=0, n_episode=0, render=0, random=False):

        assert sum([n_step!=0, n_episode!=0]) == 1, \
            'only 1 of n_step or n_episode should > 0'

        start_time = time.time()
        current_step = 0
        current_episode = 0 # np.zeros(self.env_num, dtype=int) if self._multi_env else 0
        reward_list, length_list = [], []

        if render > 0:
            render_imgs = []

        while True:
            if self._multi_env:
                batch_data = Batch(obs=self._obs)
            else:
                batch_data = Batch(obs=np.array([self._obs]))

            if random:
                self._act = np.stack([self.act_space.sample() for _ in range(self.env_num)], axis=0)
            else:
                with torch.no_grad():
                    result, self.states = self.agent(batch_data, states=self.states)
            
                if isinstance(result.act, torch.Tensor):
                    result.act = result.act.detach().cpu().numpy()
                self._act = result.act

                if self.store_keywords:
                    for k in self.store_keywords:
                        setattr(self, '_'+k, getattr(result, k).detach().cpu().numpy())
            
            if not self._multi_env:
                self._act = self._act[0]
            obs_next, self._rew, self._done, self._info = self.env.step(self._act)
            obs_next = self.agent.state_normalizer(obs_next)

            if render > 0:
                import imageio
                frame = self.env.render('rgb_array')
                render_imgs.append(frame)
                time.sleep(render)

            self.length += 1
            self.cum_reward += self._rew

            store_dict = {}
            if self._multi_env:
                for i in range(self.env_num):
                    
                    if (not random) and self.store_keywords:
                        store_dict = {k:getattr(self, '_'+k)[i] for k in self.store_keywords}

                    # TODO: add comments here
                    tmp_done = self._done[i] if not hasattr(self._info[i], 'done') else self._info[i]['done']

                    self._cached_replays[i].add({
                        'obs': self._obs[i],
                        'act': self._act[i],
                        'rew': self._rew[i],
                        'done': tmp_done,
                        'obs_next': obs_next[i],
                        **store_dict
                    })
                    current_step += 1
                    if self._done[i]:
                        current_episode += 1
                        reward_list.append(self.cum_reward[i])
                        length_list.append(self.length[i])
                        self.replay.update(self._cached_replays[i])
                        
                        self.cum_reward[i], self.length[i] = 0, 0
                        self._reset_states(i)
                        
                        self._cached_replays[i].reset()
                
                if sum(self._done) > 0:
                    obs_next = self.env.reset(np.where(self._done)[0])
                    obs_next[self._done] = self.agent.state_normalizer(obs_next[self._done])
                if n_episode != 0 and current_episode >= n_episode:
                    break        

            else:
                if self.store_keywords:
                    store_dict = {k:getattr(self, '_'+k) for k in self.store_keywords}
                    
                self.replay.add({
                    'obs': self._obs,
                    'act': self._act,
                    'rew': self._rew,
                    'done': self._done,
                    'obs_next': obs_next,
                    **store_dict
                })

                current_step += 1
                if self._done:
                    current_episode += 1
                    reward_list.append(self.cum_reward)
                    length_list.append(self.length)
                    self.cum_reward = self.length = 0
                    
                    obs_next = self.agent.state_normalizer(self.env.reset())
                    self._reset_states()

                    if n_episode != 0 and current_episode >= n_episode:
                        break
            
            if n_step != 0 and current_step >= n_step:
                break

            self._obs = obs_next
        
        self._obs = obs_next

        duration = time.time() - start_time
        # step/s and episode/s
        # step_ps = current_step / duration
        # episode_ps = current_episode / duration

        self.collect_step += current_step
        self.collect_episode += current_episode
        self.collect_time += duration
        
        n_episode = max(1, current_episode)

        if render > 0:
            imageio.mimsave('render.gif', render_imgs, 'GIF', duration=0.02)
        
        return {
            'n_epis': current_episode,
            'n_step': current_step,
            'reward': sum(reward_list) / n_episode,
            'length': sum(length_list),
            'reward_list': reward_list,
            'length_list': length_list
        }

    def sample(self, batch_size, return_indices=False):
        batch_data, indices = self.replay.sample(batch_size)
        batch_data = self.process_fn(batch_data, self.replay)
        if return_indices:
            return batch_data, indices
        else:
            return batch_data


class VirtualCollector:

    def __init__(self, agent, virtualenv, replay=None,
                    store_keywords=None, safety_task=False):
        """VirtualCollector
        collect data in virtual env model instead of real env

        args:
            agent: agent of reinforcement learning
            virtualenv: virtual env to collect
            replay: experience pool
            store_keywords: variable to be stored after each step of agent
        
        return:
            None
        """
        self.agent = agent
        self.process_fn = virtualenv.process_fn
        self.virtualenv = virtualenv
        self._multi_env = False
        self.env_num = 1
        
        self.replay = replay if replay is not None else Storage(1000)
        self.store_keywords = store_keywords
        self._store_dict = {}
        self._safety_task = safety_task

        self.states = None

        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0.0

        self.reset()
        self.reset_replay()
    

    def reset(self):
        self._obs = self._act = self._rew = None
        if self._safety_task:
            self._cost = None


    def reset_replay(self):
        self.replay.reset()


    def _reset_states(self, index=None):
        if hasattr(self.agent, 'reset_states'):
            self.agent.reset_states(self.states, index)


    def collect(self, n_step=0, init_obs=None):
        '''virtual collector cannot collect 1 whole episode.
        
        args:
            n_step: number of steps to collect
            init_obs: initial obs to start
        '''
        self.reset()

        if init_obs.ndim == 1:
            self._obs = np.array([init_obs])
            n_batch = 1
        else:
            self._obs = init_obs
            n_batch = len(init_obs)

        current_step = 0

        obses, acts, rews, obs_nexts, = [], [], [], []
        if self._safety_task:
            costs = []

        while True:
            self._store_dict.clear()
            batch_data = Batch(obs=self._obs)

            with torch.no_grad():
                result, self.states = self.agent(batch_data, states=self.states)
            
            if isinstance(result.act, torch.Tensor):
                result.act = result.act.detach().cpu().numpy()
            self._act = result.act

            if self.store_keywords:
                for k in self.store_keywords:
                    self._store_dict[k] = getattr(result, k).detach().cpu().numpy()
            
            env_returns = self.virtualenv.step(self._obs, self._act)
            obs_next = env_returns[0].detach().cpu().numpy()
            self._rew = env_returns[1].detach().cpu().numpy().squeeze()

            obses.append(self._obs)
            acts.append(self._act)
            rews.append(self._rew)
            obs_nexts.append(obs_next)
            if self._safety_task:
                self._cost = env_returns[2].detach().cpu().numpy().squeeze()
                costs.append(self._cost)

            current_step += 1
            
            if n_step != 0 and current_step >= n_step:
                # add to replay
                obses = np.stack(obses, axis=1)
                acts = np.stack(acts, axis=1)
                rews = np.stack(rews, axis=1)
                obs_nexts = np.stack(obs_nexts, axis=1)
                if self._safety_task:
                    costs = np.stack(costs, axis=1)

                dones = np.zeros(current_step)
                dones[-1] = 1.0 # TODO: 

                for i in range(n_batch):
                    data_dict = {
                        'obs': obses[i].copy(),
                        'act': acts[i].copy(),
                        'rew': rews[i].copy(),
                        'done': dones,
                        'obs_next': obs_nexts[i].copy(),
                        # 'isvirtual': 1.0,
                    }
                    if self._safety_task:
                        data_dict['cost'] = costs[i].copy()
                    self.replay.add_list(data_dict, length=current_step)
                
                break

            self._obs = obs_next
        

    def sample(self, batch_size):
        batch_data, _ = self.replay.sample(batch_size)
        return batch_data