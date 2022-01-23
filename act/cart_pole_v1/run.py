import logging

import gym
import matplotlib.pyplot as plt
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from act.cart_pole_v1.constant import DATABASE_NAME, TABLE_NAME, EXPERIENCE_REPLAY_SAVE
from src.dqn.dqn_model import DQN
from src.utils.file_utils import create_save_weight_file_path
from src.utils.sqlite_utils import create_connection, get_latest_weight, insert_data, create_table
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


def reward_function(total_reward, current_reward, terminate):
    if total_reward < 200 and terminate:
        return current_reward * -1
    return current_reward


if __name__ == '__main__':
    agent = DQN(0.9, 1, 600, 70000)
    optimizer_a = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
    agent.target_network.add(Dense(64, activation='relu', input_shape=(4,)))
    agent.target_network.add(Dense(32, activation='softmax'))
    agent.target_network.add(Dense(2, activation='linear'))
    agent.target_network.compile(optimizer=optimizer_a, loss=losses.Huber(delta=2))

    optimizer_b = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
    agent.training_network.add(Dense(64, activation='relu', input_shape=(4,)))
    agent.training_network.add(Dense(32, activation='softmax'))
    agent.training_network.add(Dense(2, activation='linear'))
    agent.training_network.compile(optimizer=optimizer_b, loss=losses.Huber(delta=2))
    episode = 1000
    env = gym.make('CartPole-v0')
    logging.debug(env.reset())
    decay_value = 0.981
    db_conn = create_connection(DATABASE_NAME)
    create_table(db_conn, TABLE_NAME)
    latest_save = get_latest_weight(db_conn, table_name=TABLE_NAME)
    if latest_save is not None:
        agent.training_network.load_weights(filepath=latest_save.weight_file)
        agent.exp_replay.load(EXPERIENCE_REPLAY_SAVE)
    agent.update_target_network()

    for i in range(episode):
        logging.debug('----------episode {}------------'.format(i))
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.observe_on_training(observation)
            observation, reward, done, _ = env.step(action)
            score += reward
            logging.debug(reward_function(score, reward, done))
            agent.take_reward(reward_function(score, reward, done), observation, done)
            agent.train_network(512, 1, 1, cer_mode=True)
            agent.update_target_network(0.0002)
            agent.epsilon_greedy.decay(decay_value, 0.01)
        if i % 5 == 0:
            file_path = create_save_weight_file_path()
            agent.training_network.save_weights(filepath=file_path)
            insert_data(file_path, i, db_conn, TABLE_NAME)
            agent.exp_replay.save(EXPERIENCE_REPLAY_SAVE)
    plt.show()
    db_conn.close()
