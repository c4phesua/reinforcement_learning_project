from unittest import TestCase
from unittest.mock import patch, MagicMock

from src.utils.epsilon_greedy import EpsilonGreedy


class TestEpsilonGreedy(TestCase):

    @patch('src.utils.epsilon_greedy.np')
    def test_perform_with_randomness(self, mock_numpy):
        mock_numpy.random.sample = MagicMock(return_value=0.1)
        mock_numpy.random.randint = MagicMock(return_value=1)
        mock_numpy.random.choice = MagicMock(return_value=2)
        model = EpsilonGreedy(0.2)
        q_value = [1, 2, 3, 4, 5]
        self.assertEqual(1, model.perform(q_value))
        self.assertEqual(2, model.perform(q_value, []))

    @patch('src.utils.epsilon_greedy.np.random')
    def test_perform_with_max_value(self, mock_numpy):
        mock_numpy.sample = MagicMock(return_value=0.21)
        model = EpsilonGreedy(0.2)
        q_value = [1, 3, 7, 3, 5, 4]
        action_space = [0, 1, 4, 5]
        self.assertEqual(2, model.perform(q_value=q_value))
        self.assertEqual(4, model.perform(q_value=q_value, action_space=action_space))

    def test_decay(self):
        model = EpsilonGreedy(0.2)
        self.assertEqual(0.2, model.epsilon)
        model.decay(0.5, 0.05)
        self.assertEqual(0.1, model.epsilon)
        model.decay(0.5, 0.05)
        self.assertEqual(0.05, model.epsilon)
        model.decay(0.5, 0.05)
        self.assertEqual(0.05, model.epsilon)
        model.decay(0.5, 0.05)
        self.assertEqual(0.05, model.epsilon)
