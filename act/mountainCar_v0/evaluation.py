import json
import logging
import pickle

import gym
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Dense

from act.mountainCar_v0.constant import EVALUATION_QUEUE_NAME
from src.dqn.dqn_model import DQN
from src.utils.rabbitmq_utils import start_consumer

env = gym.make('MountainCar-v0')


def evaluation_function(ch, method, properties, body):
    json_body = json.loads(body)
    agent = DQN(0.9, 1, 200, 70000)
    optimizer = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
    agent.training_network.add(Dense(64, activation='elu', input_shape=(2,)))
    agent.training_network.add(Dense(32, activation='elu'))
    agent.training_network.add(Dense(3, activation='linear'))
    agent.training_network.compile(optimizer=optimizer, loss=losses.Huber(delta=1))
    agent.training_network.set_weights(pickle.loads(bytes.fromhex(json_body['model_weights'])))
    logging.debug('----------episode {}------------'.format(json_body['episode']))
    avg_score = 0
    try:
        for i in range(10):
            logging.debug('----------Test {}------------'.format(i))
            observation = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.observe(observation)
                observation, reward, done, _ = env.step(action)
                score += reward
                # env.render()
            avg_score += score / 10.0
        logging.debug(f'------avg score = {avg_score}-----------')
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as ex:
        logging.exception(ex)


if __name__ == '__main__':
    start_consumer(EVALUATION_QUEUE_NAME, evaluation_function)
