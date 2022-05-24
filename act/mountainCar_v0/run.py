import logging

import gym
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses

from src.dqn.dqn_model import DQN


def reward_function(coins, state, done):
    result = 0
    if done and state[0] >= 0.6:
        return 2
    elif -0.5 < state[0] < -0.4:
        result = coins[0]
        coins[0] = 0
    elif -0.4 < state[0] < -0.3:
        result = coins[1]
        coins[1] = 0
    elif -0.3 < state[0] < -0.2:
        result = coins[2]
        coins[2] = 0
    elif -0.2 < state[0] < -0.1:
        result = coins[3]
        coins[3] = 0
    elif -0.1 < state[0] < 0:
        result = coins[4]
        coins[4] = 0
    elif 0 < state[0] < 0.1:
        result = coins[5]
        coins[5] = 0
    elif 0.1 < state[0] < 0.2:
        result = coins[6]
        coins[6] = 0
    elif 0.2 < state[0] < 0.3:
        result = coins[7]
        coins[7] = 0
    elif 0.3 < state[0] < 0.4:
        result = coins[8]
        coins[8] = 0
    elif 0.4 < state[0] < 0.5:
        result = coins[9]
        coins[9] = 0
    return result


if __name__ == '__main__':
    agent = DQN(0.9, 1, 100, 70000)
    optimizer_a = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
    agent.target_network.add(Dense(8, activation='relu', input_shape=(2,)))
    agent.target_network.add(Dense(24, activation='softmax'))
    agent.target_network.add(Dense(2, activation='linear'))
    agent.target_network.compile(optimizer=optimizer_a, loss=losses.Huber(delta=2))

    optimizer_b = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
    agent.training_network.add(Dense(8, activation='relu', input_shape=(2,)))
    agent.training_network.add(Dense(24, activation='softmax'))
    agent.training_network.add(Dense(2, activation='linear'))
    agent.training_network.compile(optimizer=optimizer_b, loss=losses.Huber(delta=2))
    episode = 1000
    env = gym.make('MountainCar-v0')
    logging.debug(env.reset())
    decay_value = 0.981
    agent.update_target_network()

    for i in range(episode):
        logging.debug('----------episode {}------------'.format(i))
        observation = env.reset()
        coins_arr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        done = False
        score = 0
        while not done:
            action = agent.observe_on_training(observation)
            observation, reward, done, _ = env.step(action)
            env.render()
            score += reward
            logging.debug((observation, reward_function(coins_arr, observation, done), done, coins_arr))
            agent.take_reward(reward_function(coins_arr, observation, done), observation, done)
            agent.train_network(64, 1, 1, cer_mode=True)
            agent.update_target_network(0.0002)
            agent.epsilon_greedy.decay(decay_value, 0.01)
    plt.show()
