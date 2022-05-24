from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Dense

from src.dqn.dqn_model import DQN


class MockSequentialA:
    @staticmethod
    def predict(arr):
        if (arr == np.array([[1, 2], [2, 3], [3, 4], [4, 5]])).all():
            return [[0.2, 0.25], [0.3, 0.35], [0.4, 0.45], [0.5, 0.55]]
        if (arr == np.array([[6, 7], [8, 9], [9, 10], [11, 12]])).all():
            return [[1.2, 1.25], [1.3, 1.35], [1.4, 1.45]]
        return np.array([[1, 10, 8, 9, 6, 5]])


class MockSequentialB:
    @staticmethod
    def predict(arr):
        return [[1.2, 1.25], [1.3, 1.35], [1.4, 1.45], [1.5, 1.55]]


class TestDqnModel(TestCase):

    @patch('src.dqn.dqn_model.Sequential')
    def test_observe_given_all_action_space_available(self, mock_sequential_model):
        mock_sequential_model.return_value = MockSequentialA()
        model = DQN(0.1, 0.5, 100, 1000)
        self.assertEqual(1, model.observe([1, 2]))

    @patch('src.dqn.dqn_model.Sequential')
    def test_observe_given_some_action_space_available_1(self, mock_sequential_model):
        mock_sequential_model.return_value = MockSequentialA()
        model = DQN(0.1, 0.5, 100, 1000)
        self.assertEqual(2, model.observe([1, 2], [0, 2, 5]))

    @patch('src.dqn.dqn_model.Sequential')
    def test_observe_given_some_action_space_available_2(self, mock_sequential_model):
        mock_sequential_model.return_value = MockSequentialA()
        model = DQN(0.1, 0.5, 100, 1000)
        self.assertEqual(3, model.observe([1, 2], [3, 2, 5, 4]))

    @patch('src.utils.epsilon_greedy.np.random')
    @patch('src.dqn.dqn_model.Sequential')
    def test_observe_on_training_given_all_action_space_available(self, mock_sequential_model, mock_numpy):
        mock_sequential_model.return_value = MockSequentialA()
        mock_numpy.sample = MagicMock(return_value=0.6)
        model = DQN(0.1, 0.5, 100, 1000)
        self.assertEqual(1, model.observe_on_training([1, 2]))
        self.assertEqual([[1, 2], 1], model.cache)

    @patch('src.utils.epsilon_greedy.np.random')
    @patch('src.dqn.dqn_model.Sequential')
    def test_observe_on_training_given_some_action_space_available(self, mock_sequential_model, mock_numpy):
        mock_sequential_model.return_value = MockSequentialA()
        mock_numpy.sample = MagicMock(return_value=0.6)
        model = DQN(0.1, 0.5, 100, 1000)
        self.assertEqual(3, model.observe_on_training([1, 2], [3, 2, 5, 4]))
        self.assertEqual([[1, 2], 3], model.cache)

    @patch('src.utils.epsilon_greedy.np.random')
    @patch('src.dqn.dqn_model.Sequential')
    def test_take_reward(self, mock_sequential_model, mock_numpy):
        mock_sequential_model.return_value = MockSequentialA()
        mock_numpy.sample = MagicMock(return_value=0.6)
        model = DQN(0.1, 0.5, 100, 1000)
        model.observe_on_training([1, 2], [3, 2, 5, 4])
        model.take_reward(0.3, [2, 1], True)
        self.assertEqual([[[1, 2], 3, 0.3, [2, 1], True]], model.exp_replay.memory)

    def test_replay(self):
        # given
        # q values [[0.2, 0.25], [0.3, 0.35], [0.4, 0.45], [0.5, 0.55]]
        # next q values [[1.2, 1.25], [1.3, 1.35], [1.4, 1.45], [1.5, 1.55]]
        model = DQN(0.1, 0.5, 100, 1000)
        model.training_network = MockSequentialA()
        model.target_network = MockSequentialB()
        states = [[1, 2], [2, 3], [3, 4], [4, 5]]
        actions = [0, 1, 0, 1]
        rewards = [0.1, 0.2, 0.3, 0.4]
        next_states = [[6, 7], [8, 9], [9, 10], [11, 12]]
        terminate = [True, True, False, False]
        states_result, q_values = model.replay(states, actions, rewards, next_states, terminate)
        expected_q_values = [[0.1, 0.25], [0.3, 0.2], [0.44499999999999995, 0.45], [0.5, 0.555]]
        self.assertEqual(states, states_result)
        self.assertListEqual(expected_q_values, q_values)

    def test_model_predict(self):
        model = DQN(0.1, 0.5, 100, 1000)
        optimizer = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
        model.target_network.add(Dense(64, activation='relu', input_shape=(2,)))
        model.target_network.compile(optimizer=optimizer, loss=losses.Huber(delta=2))
        states = [[1, 2], [2, 3], [3, 4], [4, 5], [2, 3]]
        q_values = model.target_network.predict(np.array(states))
        for i in range(4):
            predicted = model.target_network.predict(np.array([states[i]]))
            np.testing.assert_array_almost_equal(q_values[i], predicted.ravel())

    def test_update_network(self):
        model = DQN(0.1, 0.5, 100, 1000)
        optimizer_1 = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
        optimizer_2 = optimizers.RMSprop(learning_rate=0.0001, rho=0.99)
        model.target_network.add(Dense(2, activation='relu', input_shape=(2,)))
        model.target_network.compile(optimizer=optimizer_1, loss=losses.Huber(delta=2))
        model.training_network.add(Dense(2, activation='relu', input_shape=(2,)))
        model.training_network.compile(optimizer=optimizer_2, loss=losses.Huber(delta=2))
        model.training_network.set_weights([np.array([[-0.3963518, 0.09354186],
                                                      [-0.17520237, -0.15401125]], dtype=np.float32),
                                            np.array([2., 3.], dtype=np.float32)])
        model.target_network.set_weights([np.array([[-0.6971593, 0.51061773],
                                                    [0.05257487, -0.02054429]], dtype=np.float32),
                                          np.array([1., 2.], dtype=np.float32)])
        model.update_target_network(0.2)
        # 0.2 0.8
        expected = [np.array([[-0.6369978, 0.427202556],
                              [0.00701942, -0.047237682]], dtype=np.float32),
                    np.array([1.2, 2.2], dtype=np.float32)]
        for i in range(len(expected)):
            np.testing.assert_array_almost_equal(expected[i], model.target_network.get_weights()[i])
