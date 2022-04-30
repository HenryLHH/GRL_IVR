import torch
from torch import nn
import gym
from gym.spaces import Discrete, Box
import numpy as np
from collections import deque
from dmbrl.utils import tensor
import cv2

def generate_random_lines(imshape,slant,drop_length):    
    drops=[]    
    for i in range(100): ## If You want heavy rain, try increasing this        
        if slant<0:
            x= np.random.randint(slant,imshape[1])        
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-15-drop_length)
        drops.append((x,y))
    return drops
    
def add_rain(image, slant):
    imshape = image.shape
    
    drop_length=5
    drop_width=1  
    drop_color=(20,200,200) ## a shade of gray    
    rain_drops= generate_random_lines(imshape,slant,drop_length)        
    
    for rain_drop in rain_drops:        
        cv2.line(
            image,
            (rain_drop[0],rain_drop[1]),
            (rain_drop[0]+slant,rain_drop[1]+drop_length),
            drop_color,
            drop_width
        )    
        # image= cv2.blur(image,(2,2)) ## rainy view are blurry        
        brightness_coefficient = 0.5 ## rainy days are usually shady     
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
        image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)    
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    return image_RGB


class RacingNet(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()

        n_actions = action_dim[0]

        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(state_dim)

        # Estimates the parameters of a Beta distribution over actions
        self.actor_fc = nn.Sequential(nn.Linear(conv_out_size, 256), nn.ReLU(),)

        self.alpha_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())

        # Estimates the value of the state
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, x):
        x = tensor(x)
        x = self.conv(x)

        # Estimate value of the state
        value = self.critic(x)

        # Estimate the parameters of a Beta distribution over actions
        x = self.actor_fc(x)

        # add 1 to alpha & beta to ensure the distribution is "concave and unimodal" (https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return value, alpha, beta

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv(x)

        return int(np.prod(x.size()))


class CarRacingWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip=0, frame_stack=4, wait_len=1000, change_style=False, munit_model_fn=None):
        self.env = env
        super().__init__(self.env)

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.wait_len = wait_len

        self.action_space = Box(low=0, high=1, shape=(2,))
        self.observation_space = Box(low=0, high=1, shape=(frame_stack, 96, 96))

        self.frame_buf = deque(maxlen=frame_stack)

        self.t = 0
        self.last_reward_step = 0
        # self.total_reward = 0
        # self.n_episodes = 0

        self.change_style = change_style
        self.munit_model_fn = munit_model_fn
        if self.change_style:
            self.slant = np.random.randint(-5,5)

    def preprocess(self, original_action):
        original_action = original_action * 2 - 1  # map from [0, 1] to [-1, 1]
        action = np.zeros(3)
        action[0] = original_action[0]

        # Separate acceleration and braking
        action[1] = max(0, original_action[1])
        action[2] = max(0, -original_action[1])
        return action

    def postprocess(self, frame):
        # convert to grayscale
        if self.change_style:
            frame = add_rain(frame.copy(), self.slant)
            if self.munit_model_fn is not None:
                frame = self.munit_model_fn(frame)

        grayscale = np.array([0.299, 0.587, 0.114])
        observation = np.dot(frame, grayscale) / 255.0

        return observation

    # def shape_reward(self, reward):
    #     return np.clip(reward, -1.0, 1.0)

    def get_observation(self):
        return np.array(self.frame_buf)

    def reset(self):

        self.t = 0
        self.last_reward_step = 0
        # self.total_reward = 0

        first_frame = self.postprocess(self.env.reset())

        for _ in range(self.frame_stack):
            self.frame_buf.append(first_frame)

        return self.get_observation()

    def step(self, action):
        self.t += 1
        action = self.preprocess(action)

        total_reward = 0
        for _ in range(self.frame_skip + 1):
            new_frame, reward, done, info = self.env.step(action)
            # self.total_reward += reward
            # reward = self.shape_reward(reward)
            total_reward += reward

            if reward > 0:
                self.last_reward_step = self.t

        if self.t - self.last_reward_step > self.wait_len or self.t >= 1000:
            done = True

        # reward = total_reward / (self.frame_skip + 1)

        new_frame = self.postprocess(new_frame)
        self.frame_buf.append(new_frame)

        return self.get_observation(), total_reward, done, info

from tianshou.env import ShmemVectorEnv

def make_carracing_env(task, seed, training_num, test_num, mixed=False, munit_model_fn=None, **kwargs):

    if mixed:
        env = CarRacingWrapper(gym.make("CarRacing-v0"), change_style=False)
        train_envs = ShmemVectorEnv(
            [
                lambda:
                CarRacingWrapper(gym.make("CarRacing-v0"), wait_len=30, change_style=False)
                for _ in range(training_num //2)
            ] +
            [
                lambda:
                CarRacingWrapper(gym.make("CarRacing-v0"), wait_len=30, 
                    change_style=True, munit_model_fn=munit_model_fn)
                for _ in range(training_num//2)
            ]
        )
        test_envs = ShmemVectorEnv(
            [
                lambda:
                CarRacingWrapper(gym.make("CarRacing-v0"), change_style=False)
                for _ in range(test_num//2)
            ] +
            [
                lambda:
                CarRacingWrapper(gym.make("CarRacing-v0"), 
                    change_style=True, munit_model_fn=munit_model_fn)
                for _ in range(test_num//2)
            ]
        )
    else:
        env = CarRacingWrapper(gym.make("CarRacing-v0"), change_style=False)
        train_envs = ShmemVectorEnv(
            [
                lambda:
                CarRacingWrapper(gym.make("CarRacing-v0"), wait_len=30, change_style=False)
                for _ in range(training_num)
            ]
        )
        test_envs = ShmemVectorEnv(
            [
                lambda:
                CarRacingWrapper(gym.make("CarRacing-v0"), change_style=False)
                for _ in range(test_num)
            ]
        )
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
    return env, train_envs, test_envs