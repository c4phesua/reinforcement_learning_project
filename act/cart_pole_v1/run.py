from copy import deepcopy
from threading import Thread

import gym
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

from src.dqn.dqn_model import DQN

scores = list()
plt.ion()
weights = list()


def reward_function(total_reward, current_reward, terminate):
    if total_reward < 200 and terminate:
        return current_reward * -1
    return current_reward


def show_graph():
    plt.plot(list(range(len(scores))), scores, color='blue')
    plt.pause(2)


def evaluate():
    t_agent = DQN(1, 1, 200, 10000)
    optimizer = optimizers.Adam(learning_rate=0.0025)
    t_agent.training_network.add(Dense(16, activation='relu', input_shape=(4,)))
    t_agent.training_network.add(Dense(16, activation='softmax'))
    t_agent.training_network.add(Dense(2, activation='linear'))
    t_agent.training_network.compile(optimizer=optimizer, loss=losses.Huber(delta=2.0))
    t_env = gym.make('CartPole-v0')
    while True:
        if weights:
            weight = weights.pop()
            if weight is None:
                break
            test_scores = list()
            print('start testing')
            t_agent.training_network.set_weights(weight)
            for i in range(10):
                obs = t_env.reset()
                t_done = False
                t_score = 0
                while not t_done:
                    t_action = t_agent.observe(obs)
                    obs, reward, t_done, _ = t_env.step(t_action)
                    t_score += reward
                test_scores.append(score)
            scores.append(sum(test_scores) / 10)
            show_graph()


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
    episode = 500
    env = gym.make('CartPole-v0')
    print(env.reset())
    decay_value = 0.981

    thread_1 = Thread(target=evaluate)
    thread_1.daemon = True
    thread_1.start()

    for i in range(episode):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.observe_on_training(observation)
            observation, reward, done, _ = env.step(action)
            score += reward
            print(reward_function(score, reward, done))
            agent.take_reward(reward_function(score, reward, done), observation, done)
            agent.train_network(128, 1, 1, cer_mode=True)
            agent.update_target_network(0.02)
            agent.epsilon_greedy.decay(decay_value, 0.01)
        if i % 20 == 0:
            weights.append(deepcopy(agent.training_network.get_weights()))
    weights.append(None)
    plt.show()
    thread_1.join()
