from multiprocessing import Process, Queue

import gym
import matplotlib.pyplot as plt
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from src.dqn.dqn_model import DQN


class Evaluation(Process):
    def __init__(self, weights):
        super().__init__()
        self.weights: Queue = weights
        self.scores = list()

    def show_graph(self):
        plt.plot(list(range(len(self.scores))), self.scores, color='blue')
        plt.pause(2)

    def run(self) -> None:
        plt.ion()
        agent = DQN(1, 1, 200, 10000)
        optimizer = optimizers.Adam(learning_rate=0.0025)
        agent.training_network.add(Dense(16, activation='relu', input_shape=(4,)))
        agent.training_network.add(Dense(16, activation='softmax'))
        agent.training_network.add(Dense(2, activation='linear'))
        agent.training_network.compile(optimizer=optimizer, loss=losses.Huber(delta=2.0))
        env = gym.make('CartPole-v0')
        episode = 10
        while True:
            if not self.weights.empty():
                weight = self.weights.get()
                if weight is None:
                    break
                test_scores = list()
                print('start testing')
                agent.training_network.set_weights(weight)
                for i in range(episode):
                    obs = env.reset()
                    t_done = False
                    t_score = 0
                    while not t_done:
                        t_action = agent.observe(obs)
                        obs, reward, t_done, _ = env.step(t_action)
                        t_score += reward
                    test_scores.append(t_score)
                self.scores.append(sum(test_scores) / episode)
                self.show_graph()
