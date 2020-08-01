import gym
import numpy as np

def present_state_stats(state, title='Sample game state'):
    sample_state = np.array(state)
    print('{0} => shape: {1}, mean value: {2}, range: {3} to {4}'.format(
        title,
        sample_state.shape,
        sample_state.mean(),
        sample_state.min(),
        sample_state.max()
    ))

def simplify_state_representation(state):
    pass

def run_data_experiment():
    env = gym.make('Breakout-v0')
    sample_state = env.reset()
    present_state_stats(sample_state)


run_data_experiment()