#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 9 14:11:55 2017

@author: wesn2
"""

import numpy as np


class Grid_World():
    def __init__(self, world, start, goal, reward):
        self.start = start
        self.world = world
        self.goal = goal
        self.reward = reward
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
            reward = self.reward[1]
            return self.state, reward, self.done
        elif next_state == self.goal[0]:
            reward = self.reward[2]
            self.done = True
            return next_state, reward, self.done
        elif next_state == self.goal[1]:
            reward = self.reward[3]
            self.done = True
            return next_state, reward, self.done
        else:
            reward = self.reward[0]
            return next_state, reward, self.done

    def reset(self):
        self.state = self.start
        reward = 0
        self.done = False
        return self.state, reward, self.done