import gym
import numpy as np
import matplotlib.pyplot as plt


def present_state_stats(state, title='Sample game state'):
    print('{0} => shape: {1}, mean value: {2}, range: {3} to {4}'.format(
        title,
        state.shape,
        state.mean(),
        state.min(),
        state.max()
    ))
    if len(state.shape) == 3:
        channel = state.shape[2]
        fig, axs = plt.subplots(1, channel, figsize=(10, 3))
        fig.suptitle(title)
        if channel == 1:
            axs.imshow(state[:,:,0])
        else:
            for c in range(state.shape[2]):
                axs[c].imshow(state[:,:,c])
        plt.show()
    else:
        raise RuntimeError('Wrong dims')

def to_grey_scale(state):
    return np.expand_dims(np.dot(state, [0.2989, 0.5870, 0.1140]), axis=2)

def crop_state(state):
    return state[25:-10, :, :]

def normalize_state(state):
    state = state / state.max()
    return state

def sanitize_state(state):
    sanitized_state = crop_state(state)
    sanitized_state = to_grey_scale(sanitized_state)
    sanitized_state = normalize_state(sanitized_state)
    return sanitized_state

def run_data_experiment():
    env = gym.make('Breakout-v0')
    sample_state = np.array(env.reset())
    present_state_stats(sample_state, title='Raw state')
    sanitized_state = sanitize_state(sample_state)
    present_state_stats(sanitized_state, title='Sanitized state')