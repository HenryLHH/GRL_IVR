import os
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn.functional as F

from dmbrl.utils.config import Config

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device=Config.DEVICE)
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncated_normal(tensor, lower, upper, mu=0.0, sigma=1.0):
    '''truncated norm for pytorch
    '''
    pass


class Logger:
    '''
    Not used to log into tesnorboard;
    used to estimate |Q-Q^*| or |V-V^*|
    '''
    def __init__(self, collector, r_critic, c_critic, \
        critic_type, ignore_done, resample_freq = 20) -> None:
        
        self.collector = collector
        self.replay = self.collector.replay
        
        self.r_critic = r_critic
        self.c_critic = c_critic
        self.critic_type = critic_type
        self.ignore_done = ignore_done
        
        self._gamma = 0.99
        self.batch_size = 512
        self._cnt = 0
        self.resample_freq = resample_freq


        self.writer = ""
        

    def sample_traj(self, n_episode):
        self.collector.collect(n_episode=n_episode)
    

    def _get_returns(self, batch, x_type='rew'):
        if x_type == 'rew':
            returns = batch.rew.copy()
            critic = self.r_critic
        else:
            returns = batch.cost.copy()
            critic = self.c_critic
        
        for i in reversed(range(len(returns))):
            if batch.done[i] or i == len(returns) - 1:
                if self.ignore_done:
                    if self.critic_type == 'V':
                        returns[i] = critic(batch.obs[i:i+1]).mean(1, keepdim=True).squeeze().detach().cpu().numpy()
                    elif self.critic_type == 'Q':
                        returns[i] = critic(batch.obs[i:i+1], batch.act[i:i+1]).mean(1, keepdim=True).squeeze().detach().cpu().numpy()
            else:
                returns[i] += self._gamma * last
            last = returns[i]
        return returns

    def _update_returns(self, batch):
        r_returns = self._get_returns(batch, 'rew')
        c_returns = self._get_returns(batch, 'cost')
        batch.update(returns=r_returns)
        batch.update(c_returns=c_returns)


    def compute_value_error(self, n_episode=16):
        if self._cnt % self.resample_freq == 0:
            self.collector.reset_replay()
            self.sample_traj(n_episode)
        self._cnt = (self._cnt + 1) % self.resample_freq

        batch, _ = self.replay.sample(0)
        self._update_returns(batch)

        r_errs, c_errs = [], []

        with torch.no_grad():
            for b in batch.sampler(self.batch_size):
                if self.critic_type == 'V':
                    r_value_hat = self.r_critic(b.obs).mean(1, keepdim=True).squeeze().detach().cpu().numpy()
                    c_value_hat = self.c_critic(b.obs).mean(1, keepdim=True).squeeze().detach().cpu().numpy()
                elif self.critic_type == 'Q':
                    r_value_hat = self.r_critic(b.obs, b.act).mean(1, keepdim=True).squeeze().detach().cpu().numpy()
                    c_value_hat = self.c_critic(b.obs, b.act).mean(1, keepdim=True).squeeze().detach().cpu().numpy()
            
                r_errs.append(r_value_hat - b.returns)
                c_errs.append(c_value_hat - b.c_returns)

        r_errs = np.concatenate(r_errs)
        c_errs = np.concatenate(c_errs)
        
        r_err_mu, r_err_sigma = np.mean(r_errs), np.std(r_errs)
        c_err_mu, c_err_sigma = np.mean(c_errs), np.std(c_errs)

        return r_err_mu, r_err_sigma, c_err_mu, c_err_sigma

    def plot(self):
        pass