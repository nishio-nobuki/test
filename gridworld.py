#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 9 14:11:55 2017

@author: wesn2
"""

import numpy as np
import random

class Grid_World():
    def __init__(self, world, goal, reward, shapexy):
        self.world = world
        self.goal = np.array(goal)
        self.reward = reward
        self.done = False
        self.state = np.array([0, 0])
        self.shapexy = shapexy
        self.randomrew = 1.0
        self.randomtrans = 0

    def action2state(self, action, state):
        if action == 0:
            next_state = state + np.array([-1, 0])
        elif action == 1:
            next_state = state + np.array([0, 1])
        elif action == 2:
            next_state = state + np.array([1, 0])
        elif action == 3:
            next_state = state + np.array([0, -1])
        elif action == 4:
            next_state = state
        else:
            print("state has vanished!!")
        return next_state

    def step(self, action):
        #print("______________")
        #print("state", self.state)
        #print("action", action)

        next_state = self.action2state(action, self.state)

        #random movement
        #if random.random() < self.randomtrans: #if 0  no random movement
        if False:
            tmp = self.action2state(random.randint(0, 3), next_state)
            if not(-1 in tmp or self.shapexy in tmp):
                next_state = tmp

        reward = 0
        if self.world[next_state[0], next_state[1]] == 1:
            reward = reward + self.reward[1]
            #random reward
            #if random.random() < self.randomrew:
            if True:
                if all(self.state == self.goal[0]):
                    reward = reward + self.reward[2]
                if all(self.state == self.goal[1]):
                    reward = reward + self.reward[3]
            return self.state, reward, self.done
        if all(next_state == self.goal[0]):
            # random reward
            if True:
            #if random.random() < self.randomrew:
                reward = self.reward[2]
                self.done = True
                self.state = next_state
            return next_state, reward, self.done
        elif all(next_state == self.goal[1]):
            # random reward
            if True:
            #if random.random() < self.randomrew:
                reward = self.reward[3]
                self.done = True
                self.state = next_state
            return next_state, reward, self.done
        else:
            reward = self.reward[0]
            self.state = next_state
            return next_state, reward, self.done

    def reset_env(self):
        while 1:
            x = random.randint(0, self.shapexy-1)
            y = random.randint(0, self.shapexy-1)
            if self.world[y, x] == 1:
                continue
            else:
                self.state = np.array([y, x])
                break

        reward = 0
        self.done = False
        return self.state, reward, self.done