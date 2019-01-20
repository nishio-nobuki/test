#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:33:23 2017

@author: wesn2
"""

# Importing the libraries

import sys
import time
import numpy as np
import random
import copy
from collections import namedtuple
from itertools import count

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from gridworld import Grid_World
from memory import Memory, MemoryTDerror, StateMemory, ProposedMemory, MultiMemory

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'td_error', 'id'))

class Q_Learning():
    def __init__(self, mode, un=[5]*25000):
        self.modeall = mode
        self.mode = 0 #default
        '''
        self.world = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
        '''
        self.world = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             ])
        self.goal = [[1, 1], [12, 12]]  #2つまで12
        self.shapexy = 15  #gridworldのサイズ
        self.count = 0
        self.gamma = 0.9
        self.alpha = 0.1 #0.1/0.0001
        self.per_alpha_start = 0.6
        self.per_alpha = self.per_alpha_start
        self.reward_def = [-10, -100, 100, 10]  # 何もない時、壁に当たった時、ゴール1、ゴール2
        self.num_episodes = 1000 #500
        self.num_ex = 100 #50
        self.eps_start = 0.1
        self.eps = self.eps_start + 0.01
        self.batch_size = 8
        self.batch_size_per = 8
        self.memory_size = 1000
        self.beta_start = 1.0
        self.beta = self.beta_start
        self.atb_p = 0
        self.atb_len = 1  #1

        self.env = Grid_World(self.world, self.goal, self.reward_def, self.shapexy)
        self.sum_reward_mem = np.zeros(self.num_episodes)
        self.mem_avr_abtd = np.zeros(self.num_episodes)
        self.mem_max_td = np.zeros(self.num_episodes)
        self.mem_min_td = np.zeros(self.num_episodes)
        self.mem_avr_td = np.zeros(self.num_episodes)
        self.memory_b = Memory(max_size=self.memory_size)
        self.memory_td = MemoryTDerror(max_size=self.memory_size,alpha=self.per_alpha)
        self.threshold_reward = 10
        self.q_func_t = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.q_func = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.sequence_id = 0
        self.update_num = 0
        self.ave_update_num = 0
        self.i_episode = 0

        self.max_length_ts = 0

        self.t = 0

        # if agent quits the episode and restart when it arrives at the goal: 1
        self.goal_and_end = 0

        #if you adapt the  batch size of proposed method to all other method : 1
        self.adapt_batch_size = 0
        self.update_num_list = np.zeros(self.num_episodes * 50)
        self.un = self.update_num_list

        self.reward_list = []
    def select_action(self, state):
        sample = random.random()
        if np.allclose(self.q_func[state[0], state[1]],np.array([0,0,0,0,0])):
            action = random.randint(0, 4)
        elif sample > self.eps:
            q_value = self.q_func[state[0], state[1]]
            action = np.argmax(q_value)
        else:
            action = random.randint(0, 4)
        return action

    def anneal_epsilon(self):
        self.eps = self.eps - self.eps_start/self.num_episodes

    def anneal_beta(self):
        self.beta = self.beta + (1 - self.beta_start)/self.num_episodes

    def anneal_per_alpha(self):
        self.per_alpha = self.per_alpha - self.per_alpha_start/self.num_episodes
        self.memory_td.anneal_per_alpha_mem(self.per_alpha)

    def atb_weight(self,td_error):
        return (1.0 / (((abs(td_error+0.0001)**self.alpha)/self.memory_td.get_sum_absolute_TDerror()) * self.memory_b.len() + 1)) ** self.beta

    def anneal_atb_len(self,max_TDerror):
        self.atb_p = self.atb_p + (1 / self.num_episodes)
        self.atb_len = max_TDerror ** self.atb_p

    def store_er(self, t_seq):
        # store in memory b
        self.memory_b.add(t_seq)

    def store_per(self, t_seq, td_error):
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def store_proposed3(self, t_seq):
        #td_error = t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]]) \
                   #- self.q_func[t_seq.next_state[0], t_seq.next_state[1], t_seq.action]

        next_action = np.argmax(self.q_func[t_seq.next_state[0], t_seq.next_state[1]])
        td_error = t_seq.reward + self.gamma * np.max(self.q_func_t[t_seq.next_state[0], t_seq.next_state[1], next_action])\
                   - self.q_func[t_seq.state[0], t_seq.state[1], t_seq.action]
        t_seq = t_seq._replace(td_error=td_error)
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def learning(self,t_seq):
        next_action = np.argmax(self.q_func[t_seq.next_state[0], t_seq.next_state[1]])
        self.q_func[t_seq.state[0], t_seq.state[1], t_seq.action] = \
                        self.q_func[t_seq.state[0], t_seq.state[1], t_seq.action] + self.alpha \
                        * (t_seq.reward + self.gamma * np.argmax(self.q_func_t[t_seq.next_state[0], t_seq.next_state[1], next_action])
                           - self.q_func[t_seq.state[0], t_seq.state[1], t_seq.action])

    def learning_er(self):
        if self.adapt_batch_size == 1:
            for i in range(self.un[self.i_episode * 50 + self.t]):
                r = random.randint(0, self.memory_b.len() - 1)
                self.learning(self.memory_b.buffer[r])
        else:
            for i in range(self.batch_size_per):
                r = random.randint(0, self.memory_b.len() - 1)
                self.learning(self.memory_b.buffer[r])
        self.update_num = self.update_num + self.batch_size_per

    def learning_per(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        if self.adapt_batch_size == 1:
            generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.un[self.i_episode * 50 + self.t])

            batch_memory = Memory(max_size=self.un[self.i_episode * 50 + self.t])
            idx_memory = Memory(max_size=self.un[self.i_episode * 50 + self.t])
        else:
            generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size_per)

            batch_memory = Memory(max_size=self.batch_size_per)
            idx_memory = Memory(max_size=self.batch_size_per)

        generatedrand_list = np.sort(generatedrand_list)

        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(self.memory_td.buffer[idx]) ** self.per_alpha + 0.0001
                idx += 1

            batch_memory.add(self.memory_b.buffer[idx])
            idx_memory.add(idx)
        self.update_num = self.update_num + batch_memory.len()

        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, td_error, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action = np.argmax(self.q_func[next_state_b[0], next_state_b[1]])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * self.q_func_t[next_state_b[0], next_state_b[1], next_action] \
                 - self.q_func[state_b[0], state_b[1], action_b]
            self.q_func[state_b[0], state_b[1], action_b] = self.q_func[state_b[0], state_b[1], action_b] \
                                                              + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
        batch_memory.clear()
        idx_memory.clear()

    def learning_per_minib(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size_per)

        batch_memory = Memory(max_size=self.batch_size_per)
        mini_batch = Memory(max_size=self.batch_size)
        idx_memory = Memory(max_size=self.batch_size_per)

        generatedrand_list = np.sort(generatedrand_list)

        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(self.memory_td.buffer[idx]) ** self.per_alpha + 0.0001
                idx += 1

            batch_memory.add(self.memory_b.buffer[idx])
            idx_memory.add(idx)
        self.update_num = self.update_num + batch_memory.len()

        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, td_error, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action = np.argmax(self.q_func[next_state_b[0], next_state_b[1]])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * self.q_func_t[next_state_b[0], next_state_b[1], next_action] \
                 - self.q_func[state_b[0], state_b[1], action_b]
            mini_batch.add((state_b, action_b, reward_b, next_state_b, td))

        for i, (state_b, action_b, reward_b, next_state_b, td) in enumerate(mini_batch.buffer):
            self.q_func[state_b[0], state_b[1], action_b] = self.q_func[state_b[0], state_b[1], action_b] \
                                                            + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)

        mini_batch.clear()
        batch_memory.clear()
        idx_memory.clear()

    def learning_proposed3(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        batch_memory = Memory(max_size=1000)
        idx_memory = Memory(max_size=1000)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            abstd = 0
            while tmp_sum_absolute_TDerror < randnum:
                abstd = self.memory_td.buffer[idx]
                #abstd = abs(self.memory_td.buffer[idx])
                tmp_sum_absolute_TDerror += abs(abstd) ** self.per_alpha + 0.0001
                idx += 1
            id = idx
            batch_memory.add(self.memory_b.buffer[id])
            idx_memory.add(id)
            '''
            while abstd >= 0 and id <= self.memory_b.len()-1:
                batch_memory.add(self.memory_b.buffer[id])
                idx_memory.add(id)
                abstd -= 10
                id += 1
                
            
            
            while 1:
                abstd -= self.atb_len
                id -= 1
                if abstd < 0 or id < 0 or self.memory_b.buffer[idx].id != self.memory_b.buffer[id].id:
                    break
                batch_memory.add(self.memory_b.buffer[id])
                idx_memory.add(id)
            '''
            if self.memory_td.abs_max() > 0:
                #for j in range(int(round(abstd * self.max_length_ts / self.memory_td.max()))):
                for j in range(int(round(abs(abstd) * self.max_length_ts / self.memory_td.abs_max()))):
                    id -= 1
                    if id < 0 or self.memory_b.buffer[idx].id != self.memory_b.buffer[id].id:
                        break
                    batch_memory.add(self.memory_b.buffer[id])
                    idx_memory.add(id)

        self.update_num = self.update_num + batch_memory.len()
        self.update_num_list[self.i_episode*50+self.t] += batch_memory.len()
        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, td_error, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action = np.argmax(self.q_func[next_state_b[0], next_state_b[1]])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * self.q_func_t[next_state_b[0], next_state_b[1], next_action] \
                 - self.q_func[state_b[0], state_b[1], action_b]
            self.q_func[state_b[0], state_b[1], action_b] = self.q_func[state_b[0], state_b[1], action_b] \
                                                            + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
        batch_memory.clear()
        idx_memory.clear()

    def learning_proposed_minib(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size - self.multi_batch_memory.batch_memory[0].len())
        generatedrand_list = np.sort(generatedrand_list)

        mini_batch = Memory(max_size=self.batch_size)
        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            abstd = 0
            while tmp_sum_absolute_TDerror < randnum:
                abstd = self.memory_td.buffer[idx]
                # abstd = abs(self.memory_td.buffer[idx])
                tmp_sum_absolute_TDerror += abs(abstd) ** self.per_alpha + 0.0001
                idx += 1
            id = idx
            self.multi_batch_memory.batch_memory[0].add(self.memory_b.buffer[id])
            self.multi_batch_memory.idx_memory[0].add(id)
            '''
            while abstd >= 0 and id <= self.memory_b.len()-1:
                batch_memory.add(self.memory_b.buffer[id])
                idx_memory.add(id)
                abstd -= 10
                id += 1



            while 1:
                abstd -= self.atb_len
                id -= 1
                if abstd < 0 or id < 0 or self.memory_b.buffer[idx].id != self.memory_b.buffer[id].id:
                    break
                batch_memory.add(self.memory_b.buffer[id])
                idx_memory.add(id)
            '''
            if self.memory_td.max() > 0:
                for j in range(int(round(abstd * self.max_length_ts / self.memory_td.max()))):
                    id -= 1
                    if id < 0 or self.memory_b.buffer[idx].id != self.memory_b.buffer[id].id:
                        break
                    self.multi_batch_memory.batch_memory[j+1].add(self.memory_b.buffer[id])
                    self.multi_batch_memory.idx_memory[j+1].add(id)

        self.update_num = self.update_num + self.multi_batch_memory.batch_memory[0].len()
        self.update_num_list[self.i_episode * 50 + self.t] += self.multi_batch_memory.batch_memory[0].len()


        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, td_error, _) in enumerate(self.multi_batch_memory.batch_memory[0].buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action = np.argmax(self.q_func[next_state_b[0], next_state_b[1]])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * self.q_func_t[next_state_b[0], next_state_b[1], next_action] \
                 - self.q_func[state_b[0], state_b[1], action_b]
            mini_batch.add((state_b, action_b, reward_b, next_state_b, td))

        for i, (state_b, action_b, reward_b, next_state_b, td) in enumerate(mini_batch.buffer):
            self.q_func[state_b[0], state_b[1], action_b] = self.q_func[state_b[0], state_b[1], action_b] \
                                                            + self.alpha * td
            self.memory_td.assign(self.multi_batch_memory.idx_memory[0].buffer[i], td)

        # shift memory
        self.multi_batch_memory.shift_memory()

    def episode(self):
        self.sequence_id = 0

        self.count = 0

        for self.i_episode in range(self.num_episodes):
            # Initialize the Grid_World and state
            state, reward, done = self.env.reset_env()

            sum_reward = 0
            for self.t in range(50):
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                sum_reward = sum_reward + reward
                #if done:
                    #self.goal_num += 1

                if self.memory_td.len() == 0:
                    td_error = 0
                else:
                    td_error = self.memory_td.max()

                #store the sequence
                t_seq = Transition(state=state,
                                   action=action,
                                   reward=reward,
                                   next_state=next_state,
                                   td_error=td_error,
                                   id=self.sequence_id)

                if self.mode == 0:  # normal
                    self.learning(t_seq)
                elif self.mode == 1:  #experience replay
                    self.store_er(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_er()
                elif self.mode == 2: #PER
                    self.store_per(t_seq,td_error)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_per()
                elif self.mode == 7: # proposed
                    self.store_proposed3(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_proposed3()
                elif self.mode == 8: # mini batch Per
                    self.store_per(t_seq,td_error)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_per_minib()
                elif self.mode == 9: # mini batch Pro
                    self.store_proposed3(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_proposed_minib()
                else:
                    print("mode error")
                    sys.exit()
                # Move to the next state
                state = next_state
                self.q_func_t = copy.deepcopy(self.q_func)

            if self.mode in [2,4,5,6,7,8,9]:  # PER
                self.memory_td.update_TDerror(self.memory_b, self.gamma, self.q_func, self.q_func_t)

            self.sum_reward_mem[self.i_episode] = self.sum_reward_mem[self.i_episode] + sum_reward
            #self.mem_max_td[self.i_episode] = self.mem_max_td[self.i_episode] + self.memory_td.max()
            #self.mem_avr_td[self.i_episode] = self.mem_avr_td[self.i_episode] + (self.memory_td.sum() / self.memory_size)
            #self.mem_avr_abtd[self.i_episode] = self.mem_avr_abtd[self.i_episode] + (self.memory_td.abs_sum() / self.memory_size)
            #self.mem_min_td[self.i_episode] = self.mem_min_td[self.i_episode] + self.memory_td.min()

            self.sequence_id += 1
            #self.plot_q(self.q_func)
            #self.anneal_epsilon()
            if self.mode in [2,7,8,9]:
                self.anneal_per_alpha()
            if self.mode in [7,8,9]:
                self.anneal_atb_len(self.memory_td.max())

            #mode break

            if self.goal_and_end == 1:
                if sum_reward > 1250:
                    self.count += 1
                    if self.count >= 10:
                        break
                else:
                    self.count = 0

        self.reward_list.append(sum_reward)

        self.plot_q(self.q_func)
        self.plot_q_value(self.q_func)
        self.ave_update_num += self.update_num / self.num_ex
        self.update_num = 0

    def experiment(self):
        elapsed_time = []
        result_list = []
        for self.mode in self.modeall:
            if self.mode in [8,9]:
                self.multi_batch_memory = MultiMemory((self.max_length_ts + 1), self.batch_size)
                self.multi_batch_memory.clear_all_memory()
            self.sum_reward_mem = np.zeros(self.num_episodes)
            for ep_num in tqdm(range(self.num_ex)):
                start = time.time()
                self.episode()

                if ep_num == 11110:
                    self.plot_mem_b()
                    sns.distplot(self.memory_td.buffer)
                    plt.show()

                self.reset()
                elapsed_time.append(time.time() - start)
                print(elapsed_time[-1])
            print(self.ave_update_num)

            tmp = copy.deepcopy(self.reward_list)
            result_list.append(tmp)
            self.reward_list.clear()

            #return [int(round(x/self.num_ex)) for x in self.update_num_list]
            self.un = [int(round(x / self.num_ex)) for x in self.update_num_list]
            '''
            plt.subplot(2, 2, 1)
            self.plot_result(self.sum_reward_mem / self.num_ex)
            plt.legend()
            plt.title(r"sum of reward")
            '''
            self.plot_result(self.sum_reward_mem / self.num_ex)

            plt.legend()
            plt.title(r"sum of reward")
            #plt.title(r"sum of reward(max ts length:{0})".format(self.max_length_ts))

            self.max_length_ts = self.max_length_ts + 10
        '''
        plt.subplot(2, 2, 2)
        self.plot_result(self.un)
        #plt.legend()
        plt.title(r"update num")

        plt.subplot(2, 2, 3)
        self.plot_result(self.mem_max_td / self.num_ex)
        #plt.legend()
        plt.title(r"max")

        plt.subplot(2, 2, 4)
        self.plot_result(self.mem_avr_abtd / self.num_ex)
        #plt.legend()
        plt.title(r"ab_avr")
        plt.show()

        plt.subplot(1, 2, 1)
        self.plot_result(self.mem_avr_td / self.num_ex)
        # plt.legend()
        plt.title(r"avr")

        plt.subplot(1, 2, 2)
        self.plot_result(self.mem_min_td / self.num_ex)
        # plt.legend()
        plt.title(r"min")
        '''
        print("result_list")
        for num,i in enumerate(self.modeall):
            print("i", i)
            print(result_list[num])
        plt.show()

        print(sum(elapsed_time)/self.num_ex)


    def reset(self):
        self.q_func_t = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.q_func = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.memory_b.clear()
        self.memory_td.clear()
        if self.mode in [7,9]:
            self.atb_p = 0
            self.atb_len = 1
        if self.mode in [2,7,8,9]:
            self.per_alpha = self.per_alpha_start

    def plot_q(self, Qfunc):
        print('_______________________________')
        for i in range(self.shapexy):
            for j in range(self.shapexy):
                maxq = np.argmax(Qfunc[i,j,:])
                if maxq == 0:
                    symbol = '↑  '
                elif maxq == 1:
                    symbol = '→  '
                elif maxq == 2:
                    symbol = '↓  '
                elif maxq == 3:
                    symbol = '←  '
                elif maxq == 4:
                    symbol = '.  '
                if np.allclose(Qfunc[i,j,:],[0,0,0,0,0]):
                    symbol = '0  '
                if self.world[i,j] == 1:
                    symbol = '@  '
                sys.stdout.write(symbol)
            print('|')
        print('_______________________________')
        #print(self.memory_l.len())
        #print(self.memory_v.len())
        print(self.update_num)

    def plot_q_value(self, Qfunc):
        print('_______________________________')
        for i in range(self.shapexy):
            for j in range(self.shapexy):
                maxq = int(np.min(Qfunc[i,j,:]))
                sys.stdout.write('{:>6}'.format(maxq))
            print('|')
        print('_______________________________')
        #print(self.memory_l.len())
        #print(self.memory_v.len())
        print(self.update_num)

        print('_______________________________')
        for i in range(self.shapexy):
            for j in range(self.shapexy):
                maxq = int(np.max(Qfunc[i, j, :]))
                sys.stdout.write('{:>6}'.format(maxq))
            print('|')
        print('_______________________________')
        # print(self.memory_l.len())
        # print(self.memory_v.len())
        print(self.update_num)

    def plot_result(self, result):
        if self.mode == 0:  # normal
            label = 'Q Learning'
        elif self.mode == 1:  # experience replay
            label = 'Experience Replay(uniform)'
        elif self.mode == 2:  # PER
            label = 'Prioritized Experience Replay'
        elif self.mode == 7:  # proposed
            label = 'Proposal Method(TD,l={0})'.format(self.max_length_ts)
        elif self.mode == 8:  # pro||
            label = 'Prioritized Experience Replay'
        elif self.mode == 9:  # proposed
            #label = 'Proposal Method(TD,l={0})'.format(self.max_length_ts)
            label = 'l={0}'.format(self.max_length_ts)
        else:
            print("mode error")
            sys.exit()

        plt.plot(result, label=label)
        print(self.ave_update_num)

    def plot_mem_b(self):
        for (i,j) in zip(self.memory_b.buffer,self.memory_td.buffer):
            print(i)
            print(j)

# [1]損失関数の定義
# 損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

#for reshape
def rs12(state):
    return np.reshape(state, [1, 2])

# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.001, state_size=2, action_size=5, hidden_size=6):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdamとし、勾配は最大1にクリップする
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 2))
        targets = np.zeros((batch_size, 5))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(rs12(next_state_b))[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(rs12(next_state_b))[0][next_action]

            targets[i] = self.model.predict(rs12(state_b))  # Qネットワークの出力
            targets[i][action_b] = target  # 教師信号
        self.model.fit(inputs, targets, epochs=10, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


    # [※p1] 優先順位付き経験再生で重みの学習
    def prioritized_experience_replay(self, memory, batch_size, gamma, targetQN, memory_TDerror, alpha):

        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = memory_TDerror.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror,batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        batch_memory = Memory(max_size=batch_size*10)
        idx_memory = Memory(max_size=batch_size*10)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i,randnum) in enumerate(generatedrand_list):

            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(memory_TDerror.buffer[idx]) ** alpha + 0.0001
                idx += 1

            batch_memory.add(memory.buffer[idx])
            idx_memory.add(idx)


        # あとはこのバッチで学習する
        inputs = np.zeros((batch_memory.len(), 2))
        targets = np.zeros((batch_memory.len(), 5))
        for i, (state_b, action_b, reward_b, next_state_b, td_error, _) in enumerate(batch_memory.buffer):
            inputs[i:i + 1] = state_b
            target = reward_b

            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            retmainQs = self.model.predict(rs12(next_state_b))[0]
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            target = reward_b + gamma * targetQN.model.predict(rs12(next_state_b))[0][next_action]


            targets[i] = self.model.predict(rs12(state_b))  # Qネットワークの出力
            targets[i][action_b] = target  # 教師信号

        #self.model.fit(inputs, targets, batch_size=batch_memory.len(), epochs=10, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        self.model.fit(inputs, targets, epochs=10, verbose=0)

    def proposal_replay_method(self, memory, multi_batch_memory, batch_size, gamma, targetQN, memory_TDerror, max_ts_length, alpha):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = memory_TDerror.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, batch_size - multi_batch_memory.batch_memory[0].len())
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):

            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(memory_TDerror.buffer[idx]) ** alpha+ 0.0001
                idx += 1
            multi_batch_memory.batch_memory[0].add(memory.buffer[idx])
            multi_batch_memory.idx_memory[0].add(idx)

            id = idx
            #td_avr = memory_TDerror.get_avr_TDerror()
            #td_sd = memory_TDerror.get_standard_deviation(td_avr)
            # std_td = abs((memory_TDerror.buffer[id] - td_avr)/td_sd)
            #std_td = -(memory_TDerror.buffer[id] - td_avr) / td_sd

            if memory_TDerror.max() > 0:
                #for i in range(int(round(( - memory_TDerror.buffer[idx]) * max_ts_length / abs(memory_TDerror.min_under0())))):
                for i in range(int(round(memory_TDerror.buffer[idx] * max_ts_length / memory_TDerror.max()))):
                #for i in range(int(round(abs(memory_TDerror.buffer[idx]) * max_ts_length / memory_TDerror.abs_max()))):
                    id -= 1
                    if id < 0:
                        break
                    multi_batch_memory.batch_memory[i+1].add(memory.buffer[id])
                    multi_batch_memory.idx_memory[i+1].add(id)



        # あとはこのバッチで学習する
        inputs = np.zeros((multi_batch_memory.batch_memory[0].len(), 2))
        targets = np.zeros((multi_batch_memory.batch_memory[0].len(), 5))
        for i, (state_b, action_b, reward_b, next_state_b, td_error, _) in enumerate(multi_batch_memory.batch_memory[0].buffer):
            inputs[i:i + 1] = state_b
            target = reward_b

            #if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            retmainQs = self.model.predict(rs12(next_state_b))[0]
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            target = reward_b + gamma * targetQN.model.predict(rs12(next_state_b))[0][next_action]

            targets[i] = self.model.predict(rs12(state_b))  # Qネットワークの出力
            targets[i][action_b] = target  # 教師信号

        # self.model.fit(inputs, targets, batch_size=batch_memory.len(), epochs=10, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
        self.model.fit(inputs, targets, epochs=10, verbose=0)

class Deep_Q_Learning(Q_Learning):
    def __init__(self, mode, un=[5] * 25000):
        super().__init__(mode, un=[5] * 25000)

        self.q_func_t = QNetwork(learning_rate=self.alpha)  # メインのQネットワーク
        self.q_func = QNetwork(learning_rate=self.alpha)  # メインのQネットワーク


    def select_action(self, state):
        sample = random.random()
        if sample > self.eps:
            retTargetQs = self.q_func.model.predict(rs12(state))[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = random.randint(0, 4)
        return action

    def store_er(self, t_seq):
        # store in memory b
        self.memory_b.add(t_seq)

    def store_per(self, t_seq, td_error):
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def store_proposed3(self, t_seq):
        # td_error = t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]]) \
        # - self.q_func[t_seq.next_state[0], t_seq.next_state[1], t_seq.action]
        next_action = np.argmax(self.q_func.model.predict(rs12(t_seq.next_state))[0])
        td_error = t_seq.reward + self.gamma * self.q_func_t.model.predict(rs12(t_seq.next_state))[0][next_action] \
                   - self.q_func.model.predict(rs12(t_seq.state))[0][t_seq.action]
        t_seq = t_seq._replace(td_error=td_error)

        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def learning_proposed3(self):
        self.q_func.proposal_replay_method(self.memory_b, self.multi_batch_memory, self.batch_size, self.gamma,
                                           self.q_func_t, self.memory_td, self.max_length_ts, self.per_alpha)
        # shift memory
        self.multi_batch_memory.shift_memory()

    def learning_per_minib(self):
        self.q_func.prioritized_experience_replay(self.memory_b, self.batch_size, self.gamma, self.q_func_t,
                                                  self.memory_td, self.per_alpha)

    def episode(self):
        self.sequence_id = 0

        self.count = 0

        for self.i_episode in tqdm(range(self.num_episodes)):
            # Initialize the Grid_World and state
            state, reward, done = self.env.reset_env()

            sum_reward = 0
            for self.t in range(50):
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                sum_reward = sum_reward + reward
                # if done:
                # self.goal_num += 1

                if self.memory_td.len() == 0:
                    td_error = 0
                else:
                    td_error = self.memory_td.max()

                # store the sequence
                t_seq = Transition(state=state,
                                   action=action,
                                   reward=reward,
                                   next_state=next_state,
                                   td_error=td_error,
                                   id=self.sequence_id)

                if self.mode == 0:  # normal
                    self.learning(t_seq)
                elif self.mode == 1:  # experience replay
                    self.store_er(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_er()
                elif self.mode == 2:  # PER
                    self.store_per(t_seq, td_error)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_per()
                elif self.mode == 7:  # proposed
                    self.store_proposed3(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_proposed3()
                elif self.mode == 8:  # mini b per
                    self.store_proposed3(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_per_minib()
                elif self.mode == 9:  # pro mini b
                    self.store_proposed3(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_proposed3()
                else:
                    print("mode error")
                    sys.exit()
                # Move to the next state
                state = next_state
                self.q_func_t = self.q_func

            if self.mode in [2, 4, 5, 6, 7, 8, 9]:  # PER
                self.memory_td.update_DeepTDerror(self.memory_b, self.gamma, self.q_func, self.q_func_t)

            self.sum_reward_mem[self.i_episode] = self.sum_reward_mem[self.i_episode] + sum_reward
            # self.mem_max_td[self.i_episode] = self.mem_max_td[self.i_episode] + self.memory_td.max()
            # self.mem_avr_td[self.i_episode] = self.mem_avr_td[self.i_episode] + (self.memory_td.sum() / self.memory_size)
            # self.mem_avr_abtd[self.i_episode] = self.mem_avr_abtd[self.i_episode] + (self.memory_td.abs_sum() / self.memory_size)
            # self.mem_min_td[self.i_episode] = self.mem_min_td[self.i_episode] + self.memory_td.min()

            self.sequence_id += 1
            # self.plot_q(self.q_func)
            # self.anneal_epsilon()
            if self.mode in [2, 7, 8, 9]:
                self.anneal_per_alpha()
            if self.mode in [7, 8, 9]:
                self.anneal_atb_len(self.memory_td.max())

            # mode break

            if self.goal_and_end == 1:
                if sum_reward > 1250:
                    self.count += 1
                    if self.count >= 10:
                        break
                else:
                    self.count = 0

        #self.plot_q(self.q_func)
        #self.plot_q_value(self.q_func)
        self.ave_update_num += self.update_num / self.num_ex
        self.update_num = 0

    def experiment(self):
        elapsed_time = []
        for self.mode in self.modeall:
            if self.mode in [9]:
                self.multi_batch_memory = MultiMemory((self.max_length_ts + 1), self.batch_size)
                self.multi_batch_memory.clear_all_memory()
            self.sum_reward_mem = np.zeros(self.num_episodes)
            for ep_num in range(self.num_ex):
                start = time.time()
                self.episode()

                if ep_num == 11110:
                    self.plot_mem_b()
                    sns.distplot(self.memory_td.buffer)
                    plt.show()

                self.reset()
                elapsed_time.append(time.time() - start)
                print(elapsed_time[-1])
            print(self.ave_update_num)

            # return [int(round(x/self.num_ex)) for x in self.update_num_list]
            self.un = [int(round(x / self.num_ex)) for x in self.update_num_list]
            '''
            plt.subplot(2, 2, 1)
            self.plot_result(self.sum_reward_mem / self.num_ex)
            plt.legend()
            plt.title(r"sum of reward")
            '''
            self.plot_result(self.sum_reward_mem / self.num_ex)
            plt.legend()
            plt.title(r"sum of reward")
            # plt.title(r"sum of reward(max ts length:{0})".format(self.max_length_ts))

            self.max_length_ts = self.max_length_ts + 10
        '''
        plt.subplot(2, 2, 2)
        self.plot_result(self.un)
        #plt.legend()
        plt.title(r"update num")

        plt.subplot(2, 2, 3)
        self.plot_result(self.mem_max_td / self.num_ex)
        #plt.legend()
        plt.title(r"max")

        plt.subplot(2, 2, 4)
        self.plot_result(self.mem_avr_abtd / self.num_ex)
        #plt.legend()
        plt.title(r"ab_avr")
        plt.show()

        plt.subplot(1, 2, 1)
        self.plot_result(self.mem_avr_td / self.num_ex)
        # plt.legend()
        plt.title(r"avr")

        plt.subplot(1, 2, 2)
        self.plot_result(self.mem_min_td / self.num_ex)
        # plt.legend()
        plt.title(r"min")
        '''
        plt.show()

        print(sum(elapsed_time) / self.num_ex)

    def reset(self):
        self.q_func_t = QNetwork(learning_rate=self.alpha)
        self.q_func = QNetwork(learning_rate=self.alpha)
        self.memory_b.clear()
        self.memory_td.clear()
        if self.mode in [7]:
            self.atb_p = 0
            self.atb_len = 1
        if self.mode in [2, 7, 8, 9]:
            self.per_alpha = self.per_alpha_start

    def plot_q(self, Qfunc):
        print('_______________________________')
        for i in range(self.shapexy):
            for j in range(self.shapexy):
                maxq = np.argmax(Qfunc[i, j, :])
                if maxq == 0:
                    symbol = '↑  '
                elif maxq == 1:
                    symbol = '→  '
                elif maxq == 2:
                    symbol = '↓  '
                elif maxq == 3:
                    symbol = '←  '
                elif maxq == 4:
                    symbol = '.  '
                if np.allclose(Qfunc[i, j, :], [0, 0, 0, 0, 0]):
                    symbol = '0  '
                if self.world[i, j] == 1:
                    symbol = '@  '
                sys.stdout.write(symbol)
            print('|')
        print('_______________________________')
        # print(self.memory_l.len())
        # print(self.memory_v.len())
        print(self.update_num)

    def plot_q_value(self, Qfunc):
        print('_______________________________')
        for i in range(self.shapexy):
            for j in range(self.shapexy):
                maxq = int(np.min(Qfunc[i, j, :]))
                sys.stdout.write('{:>6}'.format(maxq))
            print('|')
        print('_______________________________')
        # print(self.memory_l.len())
        # print(self.memory_v.len())
        print(self.update_num)

        print('_______________________________')
        for i in range(self.shapexy):
            for j in range(self.shapexy):
                maxq = int(np.max(Qfunc[i, j, :]))
                sys.stdout.write('{:>6}'.format(maxq))
            print('|')
        print('_______________________________')
        # print(self.memory_l.len())
        # print(self.memory_v.len())
        print(self.update_num)

    def plot_result(self, result):
        if self.mode == 0:  # normal
            label = 'Q Learning'
        elif self.mode == 1:  # experience replay
            label = 'Experience Replay(uniform)'
        elif self.mode == 2:  # PER
            label = 'Prioritized Experience Replay'
        elif self.mode == 7:  # proposed
            label = 'Proposal Method(TD,l={0})'.format(self.max_length_ts)
        elif self.mode == 8:  # per
            label = 'Prioritized Experience Replay'
        elif self.mode == 9:  # proposed
            label = 'Proposal Method(TD,l={0})'.format(self.max_length_ts)
        else:
            print("mode error")
            sys.exit()

        plt.plot(result, label=label)
        print(self.ave_update_num)

    def plot_mem_b(self):
        for (i, j) in zip(self.memory_b.buffer, self.memory_td.buffer):
            print(i)
            print(j)

if __name__ == '__main__':
    '''
    modeについて
    0:normal
    1:using experience replay
    2:per
    7:proposed method
    9:proposed method(mini batch)
    '''
    modeall = [9,9,9,9]
    q_learning = Q_Learning(mode=modeall)
    q_learning.experiment()
    #print(tmp)





    #plt.legend()
    #plt.show()