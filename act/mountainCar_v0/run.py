import json
import logging
import pickle
from uuid import uuid4

import gym
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses

from act.mountainCar_v0.constant import EVALUATION_QUEUE_NAME, PROFILE_NAME
from src.dqn.dqn_model import DQN
from src.utils.rabbitmq_utils import send_message, create_evaluate_request, create_queue


def reward_function(coins, state, done):
    result = -1
    if done and state[0] >= 0.5:
        return 2
    elif -0.5 < state[0] < -0.4:
        result = coins[0]
        coins[0] = -1
    elif -0.4 < state[0] < -0.3:
        result = coins[1]
        coins[1] = -1
    elif -0.3 < state[0] < -0.2:
        result = coins[2]
        coins[2] = -1
    elif -0.2 < state[0] < -0.1:
        result = coins[3]
        coins[3] = -1
    elif -0.1 < state[0] < 0:
        result = coins[4]
        coins[4] = -1
    elif 0 < state[0] < 0.1:
        result = coins[5]
        coins[5] = -1
    elif 0.1 < state[0] < 0.2:
        result = coins[6]
        coins[6] = -1
    elif 0.2 < state[0] < 0.3:
        result = coins[7]
        coins[7] = -1
    elif 0.3 < state[0] < 0.4:
        result = coins[8]
        coins[8] = -1
    elif 0.4 < state[0] < 0.5:
        result = coins[9]
        coins[9] = -1
    return result


if __name__ == '__main__':
    agent = DQN(0.99, 1, 200, 70000)
    optimizer_a = optimizers.Adam(learning_rate=0.00025)
    agent.target_network.add(Dense(64, activation='elu', input_shape=(2,)))
    agent.target_network.add(Dense(32, activation='elu'))
    agent.target_network.add(Dense(3, activation='linear'))
    agent.target_network.compile(optimizer=optimizer_a, loss=losses.Huber(delta=1))

    optimizer_b = optimizers.Adam(learning_rate=0.00025)
    agent.training_network.add(Dense(64, activation='elu', input_shape=(2,)))
    agent.training_network.add(Dense(32, activation='elu'))
    agent.training_network.add(Dense(3, activation='linear'))
    agent.training_network.compile(optimizer=optimizer_b, loss=losses.Huber(delta=1))
    episode = 4000
    env = gym.make('MountainCar-v0')
    logging.debug(env.reset())
    decay_value = 0.99
    agent.update_target_network()

    create_queue(EVALUATION_QUEUE_NAME)
    batch_id = str(uuid4())

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
            agent.update_target_network(0.001)
            agent.epsilon_greedy.decay(decay_value, 0.001)
        send_message(EVALUATION_QUEUE_NAME,
                     message=json.dumps(create_evaluate_request(PROFILE_NAME, batch_id, i,
                                                                pickle.dumps(
                                                                    agent.training_network.get_weights()).hex())))
    plt.show()
