import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from dmbrl.network import EnsembleEnvProb
from dmbrl.utils import tensor

class VirtualEnv:
    def __init__(self, dynamics, optim, regularize_coef=[5e-5, 1e-3]):
        self.dynamics = dynamics
        self.reg_coef1, self.reg_coef2 = regularize_coef
        self.optim = optim

        if isinstance(self.dynamics, EnsembleEnvProb):
            self.num_networks = self.dynamics.ens_num
        else:
            self.num_networks = 1
            raise RuntimeError('only ensemble model is valid.')
        

    def __len__(self):
        return self.num_networks
    
    def reset(self):
        pass

    def step(self, s, a, return_ensemble=False):
        ''' step forward
        Note the input s should be noramlized,
        and the output s' is also normalized.

        args:
            return_ensemble: 
                if True, return deterministic ensemble output of s&a
                s, a: N_batch x N_ensemble x ...
        '''
        s, a = tensor(s), tensor(a)
        n_batch = s.shape[0]
        ds_logits, r_logits = self.dynamics(s, a)

        if return_ensemble:
            if s.dim() == 2:
                s = s.unsqueeze(1)
            return s + ds_logits[0], r_logits[0]
        
        if self.num_networks > 1:
            mix = D.Categorical(
                tensor(np.ones([n_batch, self.num_networks]))
            )
            ds_comp = D.Independent(D.Normal(*ds_logits), 1)
            ds_dist = D.MixtureSameFamily(mix, ds_comp)
            r_comp = D.Independent(D.Normal(*r_logits), 1)
            r_dist = D.MixtureSameFamily(mix, r_comp)

            s_next = s + ds_dist.sample()
            r = r_dist.sample()
            return s_next, r

    def process_fn(self, batch, replay):
        return batch

    def learn(self, batch, batch_size=None, repeat=1):

        for _ in range(repeat):
            nlls = []
            for b in batch.ensemble_sampler(batch_size, self.num_networks):
                # b: 3-dim, N_batch x N_ensemble x ...
                
                b.act = tensor(b.act)
                b.rew = tensor(b.rew).unsqueeze(-1)
                b.obs = tensor(b.obs)
                b.obs_next = tensor(b.obs_next)

                diff_obs = b.obs_next - b.obs
                ds_logits, r_logits = self.dynamics(b.obs, b.act)
                ds_dist = D.Normal(*ds_logits)
                r_dist = D.Normal(*r_logits)

                # train `dynamics` with MLE
                nll_loss = -ds_dist.log_prob(diff_obs).sum(-1).mean()
                nll_loss += -r_dist.log_prob(b.rew).sum(-1).mean()

                loss1, loss2 = self.dynamics.compute_regularize_loss()
                reg_loss1 = self.reg_coef1 * loss1
                reg_loss2 = self.reg_coef2 * loss2

                loss = nll_loss + reg_loss1 + reg_loss2

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                nlls.append(nll_loss.item())
            
    
        # test metrics
        index = np.random.choice(len(batch), min(1000, len(batch)))
        mini_b = batch[index]

        with torch.no_grad():
            mini_b.act = tensor(mini_b.act)
            mini_b.rew = tensor(mini_b.rew).unsqueeze(-1)
            mini_b.obs = tensor(mini_b.obs)
            mini_b.obs_next = tensor(mini_b.obs_next)

            diff_obs = mini_b.obs_next - mini_b.obs
            diff_obs = diff_obs.unsqueeze(1) # add an axis on `ense dim`
            mini_b.rew = mini_b.rew.unsqueeze(1)
            
            ds_logits, r_logits = self.dynamics(mini_b.obs, mini_b.act)
            
            obs_mse = (diff_obs - ds_logits[0]).pow(2).mean()
            r_mse = (mini_b.rew - r_logits[0]).pow(2).mean()
        

        return {
            'loss/nll': np.mean(nlls),
            'loss/reg1': reg_loss1.item(),
            'loss/reg2': reg_loss2.item(),
            'metric/obs_mse': obs_mse.item(),
            'metric/r_mse': r_mse.item(),    
        }