# 10708 Spring22 Project

The repo requires the installation of [pytorch](https://pytorch.org/), [cv2](https://pypi.org/project/opencv-python/), [tianshou](https://github.com/thu-ml/tianshou).

## CarRacing

To run the experiments of PPO, run 
```
python ppo_carracing.py --cudaid 1 --contentbased
```
The above command is for content-based policy. Remove `--contentbased` to run the experiments of full-states policy (without content extractor). 

We suggest using CUDA to accelerate content extraction and policy training.

## Pong

The RL algorithms for Pong (PPO, DQN) are implemented based on [Tianshou](https://github.com/thu-ml/tianshou).

To run the experiments of PPO, run 
```
python ppo_atari.py --device cuda:1 --contentbased
```
The above command is for content-based policy. Remove `--contentbased` to run the experiments of full-states policy (without content extractor). 

Similarly, to run the experiments of DQN, run 
```
python dqn_atari.py --device cuda:1 --contentbased
```
