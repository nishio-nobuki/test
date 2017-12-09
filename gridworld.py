#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 9 14:11:55 2017

@author: wesn2
"""

import numpy as np


class Grid_World():
    def __init__(self, space_n, world, start, goal):
        self.start = start
        self.space_n = space_n
        self.world = np.pad(world, ((1, 1), (1, 1)), "constant", constant_values=1)
        self.goal = goal
        self.reward = 0
        self.done = False
        self.state = start

    def step(self, action):

        print("aaaaa", action)

        if action.numpy() == 0:
            next_state = self.state + [0, -1]
        elif action.numpy() == 1:
            next_state = self.state + [1, 0]
        elif action.numpy() == 2:
            next_state = self.state + [0, 1]
        elif action.numpy() == 3:
            next_state = self.state + [-1, 0]
        elif action.numpy == 4:
            next_state = self.state
        else:
            print("state has vanished!!")
        if self.world[next_state[0], next_state[1]] == 1:
            self.reward = -2.0
            return self.state, self.reward, self.done
        elif next_state == self.goal:
            self.reward = 2.0
            self.done = True
            return next_state, self.reward, self.done
        else:
            self.reward = -0.1
            return next_state, self.reward, self.done

    def reset(self):
        self.state = self.start
        self.reward = 0
        self.done = False
        return self.state, self.reward, self.done