import numpy as np
import scipy.stats as stats
import torch

from dmbrl.utils import Config, tensor

def truncated_normal(size, device):
    '''Approximated truncated norm
    '''
    tmp = torch.empty(size+[4,], device=device).normal_()
    valid = (tmp < 2.0) & (tmp > -2.0)
    ind = valid.max(-1, keepdim=True)[1]
    tmp = tmp.gather(-1, ind).squeeze(-1)
    tmp.clamp_(-2.0, 2.0)
    return tmp

class Solver:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, reward_fn):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")



class CEMSolver(Solver):

    def __init__(self, max_iters, popsize, num_elites, reward_fn,
                 lower_bound=-1, upper_bound=1, epsilon=0.001, tau=0.25):
        """Creates an instance of this class.
        Arguments:
            sol_shape (int): The shape of the solution space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            lower_bound (np.array): An array of lower bounds
            upper_bound (np.array): An array of upper bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            tau (float): Controls how much of the previous mean and variance is used for the next iteration.
            next_mean = tau * old_mean + (1 - tau) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self._scale, self._bias = (self.ub-self.lb)/2, (self.ub+self.lb)/2
        self.epsilon, self.tau = epsilon, tau

        self.reward_fn = reward_fn

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the reward function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        '''X = stats.truncnorm(-2, 2)'''

        sol_shape = list(mean.shape) # [plan_horizon, act_dim]


        while (t < self.max_iters) and torch.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            tmp = torch.where(lb_dist < ub_dist, lb_dist, ub_dist)
            tmp = tmp.pow(2) / 4
            constrained_var = torch.where(var < tmp, var, tmp)
        
            '''
            # extremely slow
            x = X.rvs(size=[self.popsize, *sol_shape]) 
            x = tensor(x.astype(np.float32)) * torch.sqrt(constrained_var) + mean
            '''

            x = truncated_normal(size=[self.popsize, *sol_shape], device=Config.DEVICE)
            x = x * torch.sqrt(constrained_var) + mean

            rewards = self.reward_fn(x)

            # choose elites (largest reward), run on cpu
            idx = torch.argsort(rewards.cpu(), descending=True)[:self.num_elites] # [num_elites, ]
            idx = idx.to(device=Config.DEVICE)
            elites = x[idx]

            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)

            mean = self.tau * mean + (1 - self.tau) * new_mean
            var = self.tau * var + (1 - self.tau) * new_var

            t += 1

        return mean


    def active_explore(self, init_mean, init_var, iter_num=2):
        mean, var, t = init_mean, init_var, 0
        '''X = stats.truncnorm(-2, 2)'''

        sol_shape = list(mean.shape) # [plan_horizon, act_dim]

        while t <= iter_num:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            tmp = torch.where(lb_dist < ub_dist, lb_dist, ub_dist)
            tmp = tmp.pow(2) / 4
            constrained_var = torch.where(var < tmp, var, tmp)
        
            '''
            # extremely slow
            x = X.rvs(size=[self.popsize, *sol_shape]) 
            x = tensor(x.astype(np.float32)) * torch.sqrt(constrained_var) + mean
            '''

            x = truncated_normal(size=[self.popsize, *sol_shape], device=Config.DEVICE)
            x = x * torch.sqrt(constrained_var) + mean

            if t < iter_num:
                rewards = self.reward_fn(x)[0]

                # choose elites (largest reward), run on cpu
                idx = torch.argsort(rewards.cpu(), descending=True)[:self.num_elites] # [num_elites, ]
                idx = idx.to(device=Config.DEVICE)
                elites = x[idx]

                new_mean = torch.mean(elites, dim=0)
                new_var = torch.var(elites, dim=0)

                mean = self.tau * mean + (1 - self.tau) * new_mean
                var = self.tau * var + (1 - self.tau) * new_var
            else:
                rewards_var = self.reward_fn(x)[1]
                # choose acts with largest reward var
                idx = torch.argmax(rewards_var)

                mean = x[idx]
                break

            t += 1
        

        return mean



class CEMSolverMultiEnv(Solver):

    def __init__(self, max_iters, popsize, num_elites, reward_fn,
                 lower_bound=-1, upper_bound=1, epsilon=0.001, tau=0.25):
        """Creates an instance of this class.
        Arguments:
            sol_shape (int): The shape of the solution space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            lower_bound (np.array): An array of lower bounds
            upper_bound (np.array): An array of upper bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            tau (float): Controls how much of the previous mean and variance is used for the next iteration.
            next_mean = tau * old_mean + (1 - tau) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self._scale, self._bias = (self.ub-self.lb)/2, (self.ub+self.lb)/2
        self.epsilon, self.tau = epsilon, tau

        self.reward_fn = reward_fn

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the reward function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        '''X = stats.truncnorm(-2, 2)'''

        sol_shape = list(mean.shape) # [num_envs, plan_horizon, act_dim]


        while (t < self.max_iters) and torch.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            tmp = torch.where(lb_dist < ub_dist, lb_dist, ub_dist)
            tmp = tmp.pow(2) / 4
            constrained_var = torch.where(var < tmp, var, tmp)
        
            '''
            # extremely slow
            x = X.rvs(size=[self.popsize, *sol_shape]) 
            x = tensor(x.astype(np.float32)) * torch.sqrt(constrained_var) + mean
            '''

            x = truncated_normal(size=[self.popsize, *sol_shape], device=Config.DEVICE)
            x = x * torch.sqrt(constrained_var) + mean

            rewards = self.reward_fn(x)

            # choose elites (largest reward), run on cpu
            idx = torch.argsort(rewards.cpu(), dim=0, descending=True)[:self.num_elites] # [num_elites, num_env]
            idx = idx.to(device=Config.DEVICE)
            idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,sol_shape[1], sol_shape[2])
            elites = torch.gather(x, dim=0, index=idx)

            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)

            mean = self.tau * mean + (1 - self.tau) * new_mean
            var = self.tau * var + (1 - self.tau) * new_var

            t += 1

        return mean