import json
import logging
import pickle

import gym
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Dense

from act.mountainCar_v0.constant import EVALUATION_QUEUE_NAME, PROFILE_NAME
from src.dqn.dqn_model import DQN
from src.utils.file_utils import load_json_file
from src.utils.mongodb_utils import insert_evaluation_record, get_mongo_client
from src.utils.rabbitmq_utils import start_consumer

env = gym.make('MountainCar-v0')
configs = load_json_file('configs.json')
mongo_client = get_mongo_client()


def evaluation_function(ch, method, properties, body):
    json_body = json.loads(body)
    agent = DQN(configs['discount_factor'], configs['epsilon'], configs['e_min'], configs['e_max'])
    optimizer_b = optimizers.Adam(learning_rate=configs['optimizer']['learning_rate'])
    agent.training_network.add(Dense(64, activation='elu', input_shape=(2,)))
    agent.training_network.add(Dense(32, activation='elu'))
    agent.training_network.add(Dense(3, activation='elu'))
    agent.training_network.compile(optimizer=optimizer_b, loss=losses.Huber(delta=configs['loss_func']['delta']))
    agent.training_network.set_weights(pickle.loads(bytes.fromhex(json_body['model_weights'])))
    logging.debug('----------episode {}------------'.format(json_body['episode']))
    avg_score = 0
    total_trial = 2
    try:
        for i in range(total_trial):
            logging.debug('----------Test {}------------'.format(i))
            observation = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.observe(observation)
                observation, reward, done, _ = env.step(action)
                score += reward
                # env.render()
            avg_score += score / float(total_trial)
        logging.debug(f'------avg score = {avg_score}-----------')
        insert_evaluation_record(PROFILE_NAME, json_body['batch_id'], json_body['episode'], json_body['model_weights'],
                                 avg_score, mongo_client)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as ex:
        logging.exception(ex)


if __name__ == '__main__':
    start_consumer(EVALUATION_QUEUE_NAME, evaluation_function)
