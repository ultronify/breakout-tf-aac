import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D

class ActorCriticModel(Model):
  def __init__(self):
    pass

  def call(self, inputs, **kwargs):
    pass

def train(max_eps=5):
  env = gym.make('Breakout-v0')
  print(env.action_space.n)
  print(env.observation_space.shape)
  for eps in range(max_eps):
    done = False
    state = env.reset()
    while not done:
      action = env.action_space.sample()
      next_state, reward, done, _ = env.step(action)
      state = next_state
    print('Finished {0}'.format(eps))
  env.close()

if __name__ == '__main__':
  train()