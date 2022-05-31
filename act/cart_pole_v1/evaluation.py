import logging
import time

import gym
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from act.cart_pole_v1.constant import DATABASE_NAME, TABLE_NAME
from src.dqn.dqn_model import DQN
from src.utils.sqlite_utils import get_not_evaluate_oldest_weight, create_connection, update_data
import tensorflow as tf

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.debug("{} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.debug(e)

if __name__ == '__main__':
    sleep_time = 0.2
    agent = DQN(1, 1, 200, 10000)
    optimizer = optimizers.RMSprop(learning_rate=0.025)
    agent.training_network.add(Dense(10, activation='relu', input_shape=(4,)))
    agent.training_network.add(Dense(10, activation='relu'))
    agent.training_network.add(Dense(2, activation='linear'))
    agent.training_network.compile(optimizer=optimizer, loss=losses.Huber(delta=2))
    env = gym.make('CartPole-v0')
    episode = 10
    while True:
        db_conn = create_connection(DATABASE_NAME)
        last_save = get_not_evaluate_oldest_weight(db_conn, table_name=TABLE_NAME)
        if last_save is not None:
            sleep_time = 0.2
            weight_file = last_save.weight_file
            test_scores = list()
            logging.debug('---------start testing--------')
            agent.training_network.load_weights(weight_file)
            for i in range(episode):
                obs = env.reset()
                t_done = False
                t_score = 0
                while not t_done:
                    t_action = agent.observe(obs)
                    obs, reward, t_done, _ = env.step(t_action)
                    t_score += reward
                test_scores.append(t_score)
            average_score = sum(test_scores) / episode
            update_data(last_save.id, average_score, db_conn, table_name=TABLE_NAME)
        else:
            logging.debug('------waiting for new save-------')
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, 2.5)
        db_conn.close()
