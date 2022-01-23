import logging

import numpy as np
import random
import pickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ExperienceReplay:
    def __init__(self, e_max: int):
        if e_max <= 0:
            raise ValueError('Invalid value for memory size')
        self.e_max = e_max
        self.memory = list()
        self.index = 0

    def add_experience(self, sample: list):
        if len(sample) != 5:
            raise Exception('Invalid sample')
        if len(self.memory) < self.e_max:
            self.memory.append(sample)
        else:
            self.memory[self.index] = sample
        self.index = (self.index + 1) % self.e_max

    def sample_experience(self, sample_size: int, cer_mode: bool = False):
        samples = random.sample(self.memory, sample_size)
        if cer_mode:
            samples.append(self.memory[self.index - 1])
        # state_samples, action_samples, reward_samples, next_state_samples, done_samples
        s_batch, a_batch, r_batch, ns_batch, done_batch = map(np.array, zip(*samples))
        return s_batch, a_batch, r_batch, ns_batch, done_batch

    def get_size(self):
        return len(self.memory)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)
            logging.debug('load to memory, current size is {}'.format(len(self.memory)))
