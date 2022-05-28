import json
import logging
import pickle

import gym
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Dense

from act.mountainCar_v0.constant import EVALUATION_QUEUE_NAME
from src.dqn.dqn_model import DQN
from src.utils.rabbitmq_utils import start_consumer


def evaluation_function(ch, method, properties, body):
    json_body = json.loads(body)
    agent = DQN(0.9, 1, 200, 70000)
    optimizer_b = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
    agent.training_network.add(Dense(8, activation='relu', input_shape=(2,)))
    agent.training_network.add(Dense(24, activation='softmax'))
    agent.training_network.add(Dense(2, activation='linear'))
    agent.training_network.compile(optimizer=optimizer_b, loss=losses.Huber(delta=2))
    agent.training_network.set_weights(pickle.loads(bytes.fromhex(json_body['weights'])))
    env = gym.make('MountainCar-v0')
    logging.debug('----------episode {}------------'.format(json_body['episode']))
    observation = env.reset()
    done = False
    while not done:
        action = agent.observe(observation)
        observation, reward, done, _ = env.step(action)
        # env.render()


if __name__ == '__main__':
    start_consumer(EVALUATION_QUEUE_NAME, evaluation_function)
