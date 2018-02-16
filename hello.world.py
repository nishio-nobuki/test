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
import matplotlib.pyplot as plt

print("helllo world!!")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'td_error'))
class Test():
    def __init__(self):
        self.a = np.array([0,1,2,3,4,5,6,7,8,9])
        self.b = 10
        self.c = 15
        self.d = [np.random.normal(0,0.3) for i in range(100)]
        self.e = Transition([1,2],3,[2,3], 9, 10)
        self.l = []
        self.q_func = np.zeros(shape=(30, 30, 5))
        self.q_func[1,2,3] = 5
        self.state_memory = [[[2,3] for j in range(5)] for i in range(7)]

    def select_action(self, state):
        sample = random.random()
        if np.allclose(self.q_func[state[0], state[1]],np.array([0,0,0,0,0])):
            action = random.randint(0, 4)
        elif sample > 0.1:
            q_value = self.q_func[state[0], state[1]]
            action = np.argmax(q_value)
        else:
            action = random.randint(0, 4)
        return action

    def result(self):
        a = self.select_action([1,2])
        print(a)
        print(self.q_func[1])
        plt.plot(self.a)
        plt.show()

t=Test()
t.result()