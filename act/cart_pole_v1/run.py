from copy import deepcopy
from multiprocessing import Queue

import gym
import matplotlib.pyplot as plt
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from act.cart_pole_v1.evaluation import Evaluation
from src.dqn.dqn_model import DQN

weights_transferring = Queue()


def reward_function(total_reward, current_reward, terminate):
    if total_reward < 200 and terminate:
        return current_reward * -1
    return current_reward


if __name__ == '__main__':
    agent = DQN(1, 1, 200, 10000)
    optimizer_a = optimizers.Adam(learning_rate=0.0025)
    agent.target_network.add(Dense(16, activation='relu', input_shape=(4,)))
    agent.target_network.add(Dense(16, activation='softmax'))
    agent.target_network.add(Dense(2, activation='linear'))
    agent.target_network.compile(optimizer=optimizer_a, loss=losses.Huber(delta=2.0))

    optimizer_b = optimizers.Adam(learning_rate=0.0025)
    agent.training_network.add(Dense(16, activation='relu', input_shape=(4,)))
    agent.training_network.add(Dense(16, activation='softmax'))
    agent.training_network.add(Dense(2, activation='linear'))
    agent.training_network.compile(optimizer=optimizer_b, loss=losses.Huber(delta=2.0))
    agent.update_target_network()
    episode = 90
    env = gym.make('CartPole-v0')
    print(env.reset())
    decay_value = 0.981
    evaluate = Evaluation(weights_transferring)
    evaluate.start()

    for i in range(episode):
        print('----------episode', i, '------------')
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.observe_on_training(observation)
            observation, reward, done, _ = env.step(action)
            score += reward
            print(reward_function(score, reward, done))
            agent.take_reward(reward_function(score, reward, done), observation, done)
            agent.train_network(16, 1, 1, cer_mode=True)
            agent.update_target_network(0.02)
            agent.epsilon_greedy.decay(decay_value, 0.01)
        if i % 20 == 0:
            weights_transferring.put(deepcopy(agent.training_network.get_weights()))
    weights_transferring.put(None)
    plt.show()
    evaluate.join()
