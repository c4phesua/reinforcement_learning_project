import random
from unittest import TestCase
import numpy as np

from src.utils.experience_replay import ExperienceReplay


class TestExperienceReplay(TestCase):
    def test_add_experience(self):
        test_experience_replay = ExperienceReplay(e_max=3)
        self.assertEqual([], test_experience_replay.memory)
        test_experience_replay.add_experience([1] * 5)
        self.assertEqual([[1, 1, 1, 1, 1]], test_experience_replay.memory)
        test_experience_replay.add_experience([2] * 5)
        self.assertEqual([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]], test_experience_replay.memory)
        test_experience_replay.add_experience([3] * 5)
        self.assertEqual([[1, 1, 1, 1, 1],
                          [2, 2, 2, 2, 2],
                          [3, 3, 3, 3, 3]], test_experience_replay.memory)
        test_experience_replay.add_experience([4] * 5)
        self.assertEqual([[4, 4, 4, 4, 4],
                          [2, 2, 2, 2, 2],
                          [3, 3, 3, 3, 3]], test_experience_replay.memory)
        test_experience_replay.add_experience([5] * 5)
        self.assertEqual([[4, 4, 4, 4, 4],
                          [5, 5, 5, 5, 5],
                          [3, 3, 3, 3, 3]], test_experience_replay.memory)

    def test_sample_experience_normal_mode(self):
        random.seed(0)
        experience_replay = ExperienceReplay(e_max=4)
        experience_replay.add_experience([1, 2, 3, 4, 5])
        experience_replay.add_experience([6, 7, 8, 9, 10])
        experience_replay.add_experience([11, 12, 13, 14, 15])
        experience_replay.add_experience([16, 17, 18, 19, 20])
        s_batch, a_batch, r_batch, ns_batch, done_batch = experience_replay.sample_experience(3)
        expected_s_batch = [16, 6, 1]
        expected_a_batch = [17, 7, 2]
        expected_r_batch = [18, 8, 3]
        expected_ns_batch = [19, 9, 4]
        expected_done_batch = [20, 10, 5]
        self.assertTrue(np.array_equal(expected_s_batch, s_batch))
        self.assertTrue(np.array_equal(expected_a_batch, a_batch))
        self.assertTrue(np.array_equal(expected_r_batch, r_batch))
        self.assertTrue(np.array_equal(expected_ns_batch, ns_batch))
        self.assertTrue(np.array_equal(expected_done_batch, done_batch))
