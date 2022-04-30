import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from copy import deepcopy

from dmbrl.agent import BaseAgent
from dmbrl.data import Batch
from dmbrl.network import EnsembleCritic
from dmbrl.utils import tensor, MeanStdNormalizer

class SACAgent(BaseAgent):

    def __init__(self, actor, actor_optim,\
                    critic, critic_optim,\
                    cost_critic, cost_critic_optim,\
                    tau=0.005, # soft parameter synchronize
                    gamma=0.99, 
                    alpha=0.2, # entropy factor
                    action_range=[-1., 1.],
                    state_normalizer=MeanStdNormalizer(),
                    reward_norm=False,
                    ignore_done=False,

                    num_sample_a=1, 
                    cost_coef_para=0.1,
                    cost_lim=0.0,
                    lr_cost_coef = 3e-4,
                    logger=None,
                 ):
        
        super().__init__()

        self.actor = actor
        self.actor_optim = actor_optim

        assert isinstance(critic, EnsembleCritic), 'twin reward critic is needed.'
        self.critic, self.critic_old = critic, deepcopy(critic)
        self.critic_old.eval()
        self.critic_optim = critic_optim
        
        assert isinstance(cost_critic, EnsembleCritic), 'twin cost critic is needed.'
        self.cost_critic, self.cost_critic_old = cost_critic, deepcopy(cost_critic)
        self.cost_critic_old.eval()
        self.cost_critic_optim = cost_critic_optim


        self._tau = tau
        self._gamma = gamma
        self._alpha = alpha
        self.__eps = np.finfo(np.float32).eps.item()

        assert action_range is not None
        self._act_range = action_range
        self._act_bias = (action_range[1] + action_range[0]) / 2
        self._act_scale = (action_range[1] - action_range[0]) / 2
        self.state_normalizer = state_normalizer

        self._rew_norm = reward_norm
        self._ignore_done = ignore_done

        self.num_sample_a = num_sample_a
        self.cost_coef_para = cost_coef_para # for Constrained RL
        self.cost_lim = cost_lim
        self.lr_cost_coef = lr_cost_coef

        self.logger = logger
        self._cnt = 0
        self.log_freq = 10


    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()
        self.cost_critic.train()
        self.state_normalizer.unset_read_only()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()
        self.cost_critic.eval()
        self.state_normalizer.set_read_only()

    def save_model(self, model_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'cost_critic': self.cost_critic.state_dict(),

            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'cost_critic_optim': self.cost_critic_optim.state_dict(),
        }, f'{model_path}.model')

        with open(f'{model_path}.stats', 'wb') as f:
            pickle.dump(self.state_normalizer.state_dict(), f)
        

    def load_model(self, model_path):
        models = torch.load(f'{model_path}.model')
        self.actor.load_state_dict(models['actor'])
        self.critic.load_state_dict(models['critic'])
        self.cost_critic.load_state_dict(models['cost_critic'])

        self.actor_optim.load_state_dict(models['actor_optim'])
        self.critic_optim.load_state_dict(models['critic_optim'])
        self.cost_critic_optim.load_state_dict(models['cost_critic_optim'])

        with open(f'{model_path}.stats', 'rb') as f:
            self.state_normalizer.load_state_dict(pickle.load(f))

    def sync_weights(self):
        for o, n in zip(self.critic_old.parameters(),
                        self.critic.parameters()):
            o.data.copy_(o.data * (1-self._tau) + n.data * self._tau)
        for o, n in zip(self.cost_critic_old.parameters(),
                        self.cost_critic.parameters()):
            o.data.copy_(o.data * (1-self._tau) + n.data * self._tau)
    
    def process_fn(self, batch, replay):
        if self._rew_norm:
            if self._rew_norm == 'MeanStdNorm':
                all_rew = replay.rew[:len(replay)]
                mean, std = np.mean(all_rew), np.std(all_rew)
                if std > 1e-12:
                    batch.rew = (batch.rew - mean) / std

            elif self._rew_norm == 'MaxMinNorm':
                all_rew = replay.rew[:len(replay)]
                _max, _min = np.max(all_rew), np.min(all_rew)
                if _max > _min + 1e-4:
                    batch.rew = (batch.rew - _min) / (_max - _min)

        if self._ignore_done:
            batch.done = batch.done * 0.
        return batch

    def __call__(self, batch, states=None, obs_name='obs'):
        obs = getattr(batch, obs_name)
        logits = self.actor(obs)
        dist = torch.distributions.Normal(*logits)

        if self.training:
            x = dist.rsample() # reparameterized sampling
        else:  # deterministic output for eval
            x = logits[0]
        y = torch.tanh(x)
        act = y * self._act_scale + self._act_bias
        log_prob = dist.log_prob(x) - torch.log(
            self._act_scale * (1 - y.pow(2)) + self.__eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return Batch(
            logits=logits, act=act, dist=dist, log_prob=log_prob), None

    
    def _target_q(self, batch, value_type='rew', num_a_sample=1):
        if value_type == 'rew':
            critic = self.critic_old
        elif value_type == 'cost':
            critic = self.cost_critic_old
        
        with torch.no_grad():
            target_q_list = []

            for _ in range(num_a_sample):
                obs_next_results = self(batch, obs_name='obs_next')[0]
                a_next = obs_next_results.act
                log_p_a_next = obs_next_results.log_prob
            
                if value_type == 'rew':
                    target_q, _ = critic(batch.obs_next, a_next).min(dim=1, keepdim=True)
                    target_q = batch.rew + (1.0-batch.done) * self._gamma * target_q - self._alpha * log_p_a_next
                    target_q_list.append(target_q)

                elif value_type =='cost':
                    target_q, _ = critic(batch.obs_next, a_next).max(dim=1, keepdim=True)
                    target_q = batch.cost + (1.0-batch.done) * self._gamma * target_q
                    target_q_list.append(target_q)
            
            target_q_list = torch.stack(target_q_list, dim=0)
            if value_type == 'rew':
                target_q = torch.mean(target_q_list, dim=0)
            elif value_type == 'cost':
                target_q = torch.mean(target_q_list, dim=0) # TODO

        return target_q


    def learn(self, batch, batch_size=None):
        obs_next_results = self(batch, obs_name='obs_next')[0]
        a_next = obs_next_results.act
        
        batch.rew = tensor(batch.rew).unsqueeze(-1)
        batch.cost = tensor(batch.cost).unsqueeze(-1)
        batch.done = tensor(batch.done).unsqueeze(-1)

        weight = getattr(batch, "weight", 1.0)
        weight = tensor(weight).unsqueeze(-1)

        # apply softplus to `cost_coef_para` and get `cost_coef`
        cost_coef = np.log(1+np.exp(self.cost_coef_para))
        
        # 1. train reward critic
        # target Q(s,a) = r + (1-done) * gamma * [Q(s',a') - ...]        
        target_q_r = self._target_q(batch, 'rew', self.num_sample_a)
        current_q_r = self.critic(batch.obs, batch.act)

        td_r = current_q_r - target_q_r
        critic_loss = (td_r.pow(2).sum(dim=1) * weight).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        ''''''# estimate |Q-Q^*|
        Q_err_dict = {}
        if self._cnt % self.log_freq == 0:
            r_err_mu, r_err_sigma, c_err_mu, c_err_sigma = self.logger.compute_value_error()
            
            Q_err_dict['Q_err/r_critic'] = (r_err_mu, r_err_mu-r_err_sigma, r_err_mu+r_err_sigma)
            Q_err_dict['Q_err/c_critic'] = (c_err_mu, c_err_mu-c_err_sigma, c_err_mu+c_err_sigma)
        
        self._cnt = (self._cnt + 1) % self.log_freq

        # 2. train cost critic
        # choose MAX one when using twin critic
        target_q_c = self._target_q(batch, 'cost', self.num_sample_a)
        current_q_c = self.cost_critic(batch.obs, batch.act)

        td_c = current_q_c - target_q_c
        cost_critic_loss = (td_c.pow(2).sum(dim=1) * weight).mean()
        self.cost_critic_optim.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optim.step()
        
        self.sync_weights()

        # update weight by cost td
        #  (((td1_c + td2_c) / 2.0).pow(2) + (cost_coef * (td1_c + td2_c) / 2.0).pow(2)).pow(0.5).squeeze().cpu().data.numpy()
        new_weight = td_c.mean(dim=1).squeeze().pow(2)
        new_weight = new_weight.cpu().data.numpy()
        batch.weight = np.log(1+np.exp(-new_weight))
        

        # 3. train actor
        obs_results = self(batch)[0]
        current_q_r, _ = self.critic(batch.obs, obs_results.act).min(dim=1, keepdim=True)
        current_q_c, _ = self.cost_critic(batch.obs, obs_results.act).max(dim=1, keepdim=True)

        actor_loss = - (
            current_q_r
            - self._alpha * obs_results.log_prob
            - cost_coef * current_q_c
        ).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        qr = current_q_r.mean().item()
        qc = current_q_c.mean().item()

        # update cost coef
        grad = np.exp(self.cost_coef_para) / (1+np.exp(self.cost_coef_para))
        self.cost_coef_para += self.lr_cost_coef * grad * (qc - self.cost_lim)
        # cost_coef = np.clip(cost_coef, 0.0, max_cost_coef)
        cost_coef = np.log(1+np.exp(self.cost_coef_para))

        
        return {
            'loss/actor_rew': qr,
            'loss/actor_cost': qc, 
            'loss/critic_r': critic_loss.item(),
            'loss/critic_c': cost_critic_loss.item(),
            'cost_coef': cost_coef,
            **Q_err_dict,
        }