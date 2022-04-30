import torch
import numpy as np
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from dmbrl.utils import tensor

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class EnsembleLinear(nn.Module):
    def __init__(self, num_ensemble, in_features, out_features, bias=True):
        '''2x faster than `nn.ModuleList([nn.Linear() for _ in range(num_ensemble)])`
        '''
        super(EnsembleLinear, self).__init__()
        self.num_ensemble = num_ensemble
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_ensemble, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_ensemble, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming uniform for `self.weight`
        fan_in = self.in_features
        a = np.sqrt(5)

        gain = np.sqrt(2.0 / (1 + a ** 2))
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return input.matmul(self.weight) + self.bias

    def extra_repr(self):
        return 'num_ensemble={}, in_features={}, out_features={}, bias={}'.format(
            self.num_ensemble, self.in_features, self.out_features, self.bias is not None
        )



class EnvProb(nn.Module):
    def __init__(self, s_dim, a_dim, r_dim=1, c_dim=0, layer_num=5, hidden_dim=200):
        super().__init__()
        assert layer_num >= 2, 'layer_num should be larger than 2'

        self.model = [nn.Linear(s_dim+a_dim, hidden_dim),Swish()]
        for _ in range(layer_num - 2):
            self.model += [nn.Linear(hidden_dim, hidden_dim), Swish()]
        
        self.model = nn.Sequential(*self.model)
        self.mu_net = nn.Linear(hidden_dim, s_dim+r_dim+c_dim)
        self.sigma_net = nn.Linear(hidden_dim, s_dim+r_dim+c_dim)

    def forward(self, s, a, retun_log=True):
        s, a = tensor(s), tensor(a)
        inputs = torch.cat([s,a],dim=-1)
        logits = self.model(inputs)
        mu, sigma = self.mu_net(logits), self.sigma_net(logits)
        if not retun_log:
            sigma = torch.exp(sigma)
        return (mu, sigma)

    def compute_l2_loss(self):
        l2_loss = 0.
        for name, w in self.named_parameters():
            if "weight" in name: # exclude `bias`
                l2_loss += torch.sum(torch.pow(w, 2))
                
        return l2_loss


class EnsembleEnvProb(nn.Module):
    def __init__(
        self, ens_num,
        s_dim, a_dim, r_dim=1, c_dim=0,
        layer_num=5, hidden_dim=200,
    ):
        super().__init__()
        self.ens_num = ens_num
        self.layer_num = layer_num
        self.s_dim = s_dim
        self.has_cost = c_dim > 0

        self.networks = nn.ModuleList([EnvProb(s_dim, a_dim, r_dim, c_dim, layer_num, hidden_dim)
            for _ in range(ens_num)])
        
        # constrain log var of `diff_s` & `r`
        # log var = 2 * log std
        self.max_logvar_diff_s = nn.Parameter(torch.ones([1, s_dim]) / 2.0)
        self.min_logvar_diff_s = nn.Parameter(torch.ones([1, s_dim]) * -10.0)
        self.max_logvar_r = nn.Parameter(torch.ones([1, r_dim+c_dim]) / 2.0)
        self.min_logvar_r = nn.Parameter(torch.ones([1, r_dim+c_dim]) * -10.0)
        
        self.state_normalizer = None
        self.action_normalizer = None


    def forward(self, s, a):
        '''
        args:
            s, a: 2 or 3 dim, N_batch x (N_ensemble x) ...
        return:
            mu, sigma: 3-dim, N_batch x N_ensemble x ...
        '''
        mus, sigmas = [], []
        if s.dim() == 2:
            for net in self.networks:
                mu, sigma = net(s, a)
                mus.append(mu)
                sigmas.append(sigma)
        
        elif s.dim() == 3:
            for _, net in enumerate(self.networks):
                mu, sigma = net(s[:, _], a[:, _])
                mus.append(mu)
                sigmas.append(sigma)
        
        mus, sigmas = torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)
        diff_s_mu, r_mu = mus[..., :self.s_dim], mus[..., self.s_dim:]
        diff_s_sigma, r_sigma = sigmas[..., :self.s_dim], sigmas[..., self.s_dim:]

        diff_s_sigma = (self.max_logvar_diff_s - F.softplus(self.max_logvar_diff_s - 2*diff_s_sigma)) / 2
        diff_s_sigma = (self.min_logvar_diff_s + F.softplus(2*diff_s_sigma - self.min_logvar_diff_s)) / 2
        diff_s_sigma = torch.exp(diff_s_sigma)
        r_sigma = (self.max_logvar_r - F.softplus(self.max_logvar_r - 2*r_sigma)) / 2
        r_sigma = (self.min_logvar_r + F.softplus(2*r_sigma - self.min_logvar_r)) / 2
        r_sigma = torch.exp(r_sigma)

        # if self.has_cost:
        #     r_mu, c_mu = r_mu[..., :self.r_dim], r_mu[..., self.r_dim:]
        #     r_sigma, c_sigma = r_sigma[..., :self.r_dim], r_sigma[..., self.r_dim:]

        #     return (diff_s_mu, diff_s_sigma), (r_mu, r_sigma), (c_mu, c_sigma)
        
        return (diff_s_mu, diff_s_sigma), (r_mu, r_sigma)


    def compute_regularize_loss(self):        
        loss1 = 0.
        for net in self.networks:
            loss1 += net.compute_l2_loss()
        
        loss2 = (self.max_logvar_diff_s - self.min_logvar_diff_s).sum()
        loss2 += (self.max_logvar_r - self.min_logvar_r).sum()

        return loss1, loss2


class VirtualEnv:
    def __init__(self, dynamics, optim, regularize_coef=[5e-5, 1e-3]):
        self.dynamics = dynamics
        self.has_cost = self.dynamics.has_cost
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
            if self.has_cost:
                r_mu, c_mu = r_logits[0][..., :1], r_logits[0][..., 1:]
                return s + ds_logits[0], r_mu, c_mu
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
            if self.has_cost:
                r, c = r[..., :1], r[..., 1:]
                return s_next, r, c

            return s_next, r

    def process_fn(self, batch):
        return batch

    def learn(self, batch, batch_size=None, repeat=1):

        nll_losses, reg_losses1, reg_losses2 = [], [], []
        r_mses, obs_mses = [], []

        for _ in range(repeat):
            nlls = []
            for b in batch.ensemble_sampler(batch_size, self.num_networks):
                # b: 3-dim, N_batch x N_ensemble x ...
                
                b.act = tensor(b.act)
                b.rew = tensor(b.rew).unsqueeze(-1)
                b.obs = tensor(b.obs)
                b.obs_next = tensor(b.obs_next)

                if self.has_cost:
                    b.cost = tensor(b.cost).unsqueeze(-1)

                diff_obs = b.obs_next - b.obs
                ds_logits, r_logits = self.dynamics(b.obs, b.act)
                ds_dist = D.Normal(*ds_logits)
                r_dist = D.Normal(*r_logits)

                # train `dynamics` with MLE
                nll_loss = -ds_dist.log_prob(diff_obs).mean()
                if self.has_cost:
                    nll_loss += -r_dist.log_prob(torch.cat([b.rew, b.cost], dim=-1)).mean()
                else:
                    nll_loss += -r_dist.log_prob(b.rew).mean()

                loss1, loss2 = self.dynamics.compute_regularize_loss()
                reg_loss1 = self.reg_coef1 * loss1
                reg_loss2 = self.reg_coef2 * loss2

                loss = nll_loss + reg_loss1 + reg_loss2

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                nlls.append(nll_loss.item())
            
        nll_losses.append(np.mean(nlls))
        reg_losses1.append(reg_loss1.item())
        reg_losses2.append(reg_loss2.item())
    
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
            if not self.has_cost:
                r_mse = (mini_b.rew - r_logits[0]).pow(2).mean()
            else:
                mini_b.cost = tensor(mini_b.cost).unsqueeze(-1)
                mini_b.cost = mini_b.cost.unsqueeze(1)
                r_mse = (mini_b.rew - r_logits[0][...,:1]).pow(2).mean()
                c_mse = (mini_b.cost - r_logits[0][...,1:]).pow(2).mean()
        
        '''r_mses.append(r_mse.item())
        obs_mses.append(obs_mse.item())'''
        return_dict = {
            'loss/nll': nll_losses,
            'loss/reg1': reg_losses1,
            'loss/reg2': reg_losses2,
            'metric/obs_mse': obs_mse.item(),
            'metric/r_mse': r_mse.item(),    
        }
        if self.has_cost:
            return_dict['metric/c_mse'] = c_mse.item()

        return return_dict


class EnsembleEnvProb2(nn.Module):
    def __init__(self, ens_num, layer_num, s_dim, preproc_s_dim, a_dim, hidden_dim=128):
        super().__init__()
        self.n_ensemble = ens_num
        self.layer_num = layer_num

        self.model =  [
            EnsembleLinear(ens_num, np.prod(preproc_s_dim) + np.prod(a_dim), hidden_dim),
            Swish()]

        for _ in range(layer_num):
            self.model += [EnsembleLinear(ens_num, hidden_dim, hidden_dim), Swish()]
        self.model = nn.Sequential(*self.model)

        self.mu = EnsembleLinear(ens_num, hidden_dim, np.prod(s_dim))
        self.sigma = EnsembleLinear(ens_num, hidden_dim, np.prod(s_dim))
        
        # requires grad
        self.max_logvar = nn.Parameter(torch.ones(size=(1, np.prod(s_dim)), dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(torch.ones(size=(1, np.prod(s_dim)), dtype=torch.float32) * -10.0)
        


    def forward(self, s, a):
        # s: [n_ense, batch_size, obs_dim]        
        s, a = tensor(s), tensor(a)
        inputs = torch.cat([s,a],dim=-1)
        logits = self.model(inputs)
        mu = self.mu(logits)
        sigma = self.sigma(logits)
        
        sigma = (self.max_logvar - F.softplus(self.max_logvar - 2*sigma)) / 2
        sigma = (self.min_logvar + F.softplus(2*sigma - self.min_logvar)) / 2
        sigma = torch.exp(sigma)
        
        return mu, sigma

    def regularize_loss(self, lambdas=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]):
        """ lambdas: weights for regularization loss
        assert len(lambdas) == self.lay_num
        """
        loss = 0.
        for _ in range(len(lambdas) - 1):
            para = self.model[2*_].weight
            loss += lambdas[_] * para.pow(2).sum() / 2.0
        
        loss += lambdas[-1] * (self.mu.weight.pow(2).sum() + self.sigma.weight.pow(2).sum()) / 2.0
        return loss