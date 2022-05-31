import json
import logging
import pickle

import gym
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Dense

from act.mountainCar_v0.constant import EVALUATION_QUEUE_NAME, PROFILE_NAME
from src.dqn.dqn_model import DQN
from src.utils.file_utils import load_json_file
from src.utils.mongodb_utils import create_new_batch, insert_loss
from src.utils.rabbitmq_utils import send_message, create_evaluate_request, create_queue

configs = load_json_file('configs.json')


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
    agent = DQN(configs['discount_factor'], configs['epsilon'], configs['e_min'], configs['e_max'])
    optimizer_a = optimizers.RMSprop(learning_rate=configs['optimizer']['learning_rate'],
                                     rho=configs['optimizer']['rho'])
    agent.target_network.add(Dense(8, activation='relu', input_shape=(2,)))
    agent.target_network.add(Dense(24, activation='softmax'))
    agent.target_network.add(Dense(2, activation='linear'))
    agent.target_network.compile(optimizer=optimizer_a, loss=losses.Huber(delta=2))

    optimizer_b = optimizers.RMSprop(learning_rate=configs['optimizer']['learning_rate'],
                                     rho=configs['optimizer']['rho'])
    agent.training_network.add(Dense(8, activation='relu', input_shape=(2,)))
    agent.training_network.add(Dense(24, activation='softmax'))
    agent.training_network.add(Dense(2, activation='linear'))
    agent.training_network.compile(optimizer=optimizer_b, loss=losses.Huber(delta=2))
    episode = configs['total_episode']
    env = gym.make('MountainCar-v0')
    logging.debug(env.reset())
    decay_value = configs['decay_value']
    agent.update_target_network()

    create_queue(EVALUATION_QUEUE_NAME)
    batch_id = create_new_batch(PROFILE_NAME, configs)

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
            log = agent.train_network(configs['batch_size'], 1, 1, cer_mode=True)
            if log:
                insert_loss(PROFILE_NAME, batch_id, log[0])
            agent.update_target_network(configs['tau'])
            agent.epsilon_greedy.decay(decay_value, configs['min_epsilon'])
        send_message(EVALUATION_QUEUE_NAME,
                     message=json.dumps(create_evaluate_request(PROFILE_NAME, batch_id, i,
                                                                pickle.dumps(
                                                                    agent.training_network.get_weights()).hex())))
    plt.show()
