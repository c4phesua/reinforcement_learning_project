import gym
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from act.cart_pole_v1.run import DATABASE_NAME, TABLE_NAME
from src.dqn.dqn_model import DQN
from src.utils.sqlite_utils import get_not_evaluate_latest_weight, create_connection, update_data

if __name__ == '__main__':
    agent = DQN(1, 1, 200, 10000)
    optimizer = optimizers.Adam(learning_rate=0.0025)
    agent.training_network.add(Dense(16, activation='relu', input_shape=(4,)))
    agent.training_network.add(Dense(16, activation='softmax'))
    agent.training_network.add(Dense(2, activation='linear'))
    agent.training_network.compile(optimizer=optimizer, loss=losses.Huber(delta=2.0))
    env = gym.make('CartPole-v0')
    episode = 10
    db_conn = create_connection(DATABASE_NAME)
    while True:
        last_save = get_not_evaluate_latest_weight(db_conn, table_name=TABLE_NAME)
        if last_save is not None:
            weight_file = last_save.weight_file
            test_scores = list()
            print('start testing')
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
