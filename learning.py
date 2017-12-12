#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:33:23 2017

@author: wesn2
"""

# Importing the libraries

import numpy as np
import random
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt

from gridworld import Grid_World

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'td_error'))

class Q_Learning():
    def __init__(self):
        self.gamma = 0.95
        self.alpha = 0.01
        self.start = [6,6]
        self.goal = [[1,28],[27,1]]
        self.reward_def = [-0.1, -4, 5, 7]   #何もない時、壁に当たった時、ゴール1、ゴール2
        self.num_episodes = 100
        self.eps = 0.1
        self.tau = 1.5
        self.world = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          ])
        self.env = Grid_World(self.world, self.start, self.goal, self.reward_def)
        self.episode_durations = []
        self.memory_b = []
        self.memory_t = []
        self.memory_l = []
        self.memory_v = []
        self.size_of_memb = 1000
        self.size_of_memt = 1000
        self.size_of_meml = 1000
        self.size_of_memv = 100
        self.threshold_reward = 1
        self.q_func = np.zeros(shape=(30, 30, 5))


    def select_action(self, state):
        sample = random.random()
        if sample > self.eps:
            q_value = self.qfunc(state[0],state[1]).data
            _, action = q_value.max(2)
        else:
            action = random.randint(0, 4)
        return action

    def store(self, t_seq):
        # calc TD-error
        td_error = t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0],t_seq.next_state[1]]) \
                   - self.q_func[t_seq.next_state[0],t_seq.next_state[1],t_seq.action]
        t_seq.td_error = abs(td_error)

        #store in memory b
        if len(self.memory_b) > self.size_of_memb:
            self.memory_b.pop(0)
        self.memory_b.append(t_seq)

        #store in memory t
        self.memory_t = []
        if t_seq.reward > self.threshold_reward:
            for i in range(self.size_of_memt):
                self.memory_t[i] = self.memory_b[-(i+1)]
                if self.memory_t[i].state == self.start or i > self.size_of_memb - 2:
                    break

        #store in memory l
        w_new = max(self.memory_t,key=(lambda x: x.td_error))
        w_max = 0
        for i in range(len(self.memory_l)):
            if w_max < max(self.memory_l[i],key=(lambda x: x.td_error)).td_error:
                w_max = max(self.memory_l[i],key=(lambda x: x.td_error)).td_error
        if w_new.td_error * self.tau > w_max:
            if self.memory_l.length > self.size_of_meml:
                self.memory_l.pop(0)
            self.memory_l.append(self.memory_t)

        # make virtual sequence
        self.make_virtual_sequence()

    def make_virtual_sequence(self):
        for seq in self.memory_l:
            for tra in seq:
                for trab in self.memory_b:
                    if tra.state == trab.state:
                        intersec = tra.state

    def learning(self,t_seq):
        self.q_func[t_seq.next_state[0], t_seq.next_state[1], t_seq.action] = \
                        self.q_func[t_seq.next_state[0], t_seq.next_state[1], t_seq.action] + self.alpha \
                        * (t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]])
                           - self.q_func[t_seq.next_state[0], t_seq.next_state[1], t_seq.action])

    def episode(self):
        for i_episode in range(self.num_episodes):
            # Initialize the Grid_World and state
            state, reward, done = self.env.reset()
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                #store the sequence
                t_seq = Transition(state = state,
                                   action = action,
                                   next_state = next_state,
                                   reward = reward)
                #self.store(t_seq)

                #learning
                self.learning(t_seq)

                if done:
                    self.episode_durations.append(t + 1)
                    break
                # Move to the next state
                state = next_state


if __name__ == '__main__':
    q_learning = Q_Learning()
    q_learning.episode()