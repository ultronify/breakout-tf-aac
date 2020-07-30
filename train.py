import gym
import numpy as np
import tensorflow as tf
from tensorflow import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten


class ActorCriticModel(Model):
    def __init__(self, state_shape, action_space_size):
        super(ActorCriticModel, self).__init__()
        self.state_shape = state_shape
        self.action_space_state = action_space_size
        # Image feature extraction layers
        self.conv_input = Conv2D(
            4, (3, 3), (2, 2), activation='relu', input_shape=self.state_shape)
        self.conv_0 = Conv2D(8, (3, 3), (2, 2))
        self.bn_0 = BatchNormalization()
        self.flatten_layer = Flatten()
        # Actor layers
        self.actor_dense_0 = Dense(128, activation='relu')
        self.actor_dense_1 = Dense(64, activation='relu')
        self.actor_output = Dense(
            self.action_space_state, activation='softmax')
        # Critic layers
        self.critic_dense_0 = Dense(128, activation='relu')
        self.critic_dense_1 = Dense(64, activation='relu')
        self.critic_output = Dense(1, activation='linear')

    def call(self, inputs, **kwargs):
        # Feature extraction
        x = self.conv_input(inputs)
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = self.flatten_layer(x)
        # Actor network
        p = self.actor_dense_0(x)
        p = self.actor_dense_1(p)
        p = self.actor_output(p)
        # Critic network
        v = self.critic_dense_0(x)
        v = self.critic_dense_1(v)
        v = self.critic_output(v)
        return p, v


def sample_action(action_space_size, probs, use_max=False):
    if use_max:
        return np.argmax(probs)
    else:
        return np.random.choice(action_space_size, p=probs/probs.sum())


def compute_discounted_rewards(rewards, gamma):
    discounted_reward = 0
    discounted_rewards = []
    for reward in rewards[::-1]:
        discounted_reward = gamma * discounted_reward + reward
        discounted_rewards.append([discounted_reward])
    return discounted_rewards[::-1]


def eval(model, env, max_eps, action_space_size):
    total_reward = 0.0
    for _ in range(max_eps):
        done = False
        state = env.reset()
        while not done:
            action_dist, _ = model(tf.convert_to_tensor([state]))
            action = sample_action(
                action_space_size, action_dist.numpy()[0], use_max=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
    avg_reward = total_reward / max_eps
    return avg_reward


def train(max_eps=5, gamma=0.99):
    env = gym.make('Breakout-v0')
    eval_env = gym.make('Breakout-v0')
    state_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    print('Construct model with action space size {0} and state shape {1}'.format(
        action_space_size, state_shape))
    model = ActorCriticModel(state_shape, action_space_size)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    for eps in range(max_eps):
        done = False
        state = env.reset()
        actions, rewards, states = [], [], []
        while not done:
            action_dist, _ = model(tf.convert_to_tensor(
                [state], dtype=tf.float32))
            action = sample_action(action_space_size, action_dist.numpy()[0])
            next_state, reward, done, _ = env.step(action)
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            state = next_state
        # Calculate loss and gradients
        with tf.GradientTape() as tape:
            probs_raw, vals = model(
                tf.convert_to_tensor(states, dtype=tf.float32))
            probs = tf.clip_by_value(probs_raw, 1e-10, 1-1e-10)
            log_probs = tf.math.log(probs)
            q_vals = tf.convert_to_tensor(
                compute_discounted_rewards(rewards, gamma), dtype=tf.float32)
            action_onehot = tf.one_hot(actions, action_space_size)
            advantage = q_vals - vals
            val_loss = advantage ** 2
            entropy_loss = -tf.reduce_sum(probs, log_probs)
            policy_loss = -(log_probs * action_onehot) * advantage
            loss = 0.5 * tf.reduce_mean(val_loss) + \
                tf.reduce_mean(policy_loss) + 0.01 * entropy_loss
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        eval_score = eval(model, eval_env, 10, action_space_size)
        print(
            'Finished training {0}/{1} with score {2}'.format(eps, max_eps, eval_score))
    env.close()

