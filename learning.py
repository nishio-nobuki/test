#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:33:23 2017

@author: wesn2
"""

# Importing the libraries

import sys
import numpy as np
import random
import copy
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
from tqdm import tqdm
from gridworld import Grid_World
from memory import Memory, MemoryTDerror, StateMemory, ProposedMemory

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'td_error', 'id'))

class Q_Learning():
    def __init__(self, mode):
        self.mode = mode
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
             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
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
        self.goal = [[1, 1], [1, 1]]
        self.shapexy = 15
        self.gamma = 0.9
        self.alpha = 0.1
        self.per_alpha = 0.8
        self.reward_def = [-10, -100, 100, 100]  # 何もない時、壁に当たった時、ゴール1、ゴール2
        self.num_episodes = 500
        self.num_ex = 1
        self.eps_start = 0.1
        self.eps = self.eps_start + 0.01
        self.tau = 2.0
        self.batch_size = 10
        self.batch_size_per = 25
        self.memory_size = 1000
        if self.mode == 3:
            self.memory_size_b = 250
            self.memory_size_t = 50
            self.memory_size_l = 10
            self.memory_size_v = 50
            self.memory_t = Memory(max_size=self.memory_size_t)
            self.memory_l = Memory(max_size=self.memory_size_l)
            self.memory_v = Memory(max_size=self.memory_size_v)
            self.memory_vs = Memory(max_size=self.memory_size_t)
            self.memory_vs2 = Memory(max_size=self.memory_size_t)
            self.memory_state = StateMemory(self.shapexy, self.shapexy)
        elif self.mode == 5:
            self.memory_state = ProposedMemory(self.shapexy, self.shapexy)
            self.lenv = 10
            self.memory_v = Memory(max_size=self.lenv)
        elif self.mode == 6:
            self.memory_state = ProposedMemory(self.shapexy, self.shapexy)
            self.lenv = 10
            self.memory_v = Memory(max_size=self.lenv)
            self.memory_size_t = 50
            self.memory_t = Memory(max_size=self.memory_size_t)
        self.env = Grid_World(self.world, self.goal, self.reward_def, self.shapexy)
        self.sum_reward_mem = np.zeros(self.num_episodes)
        self.memory_b = Memory(max_size=self.memory_size)
        self.memory_td = MemoryTDerror(max_size=self.memory_size,alpha=self.per_alpha)
        self.threshold_reward = 10
        self.q_func_t = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.q_func = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.sequence_id = 0
        self.goal_num = 0

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

    def store_er(self, t_seq):
        # store in memory b
        self.memory_b.add(t_seq)

    def store_per(self, t_seq, td_error):
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def store_proposed3(self, t_seq):
        td_error = t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]]) \
                   - self.q_func[t_seq.next_state[0], t_seq.next_state[1], t_seq.action]
        t_seq = t_seq._replace(td_error=abs(td_error))
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def store_ts(self, t_seq):
        # calc TD-error
        td_error = t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0],t_seq.next_state[1]]) \
                   - self.q_func[t_seq.next_state[0],t_seq.next_state[1],t_seq.action]
        t_seq = t_seq._replace(td_error=abs(td_error))
        #store in memory b
        self.memory_b.add(t_seq)

        #store in memory t
        self.memory_t.clear()
        if t_seq.reward > self.threshold_reward:
            for i in range(self.memory_size_t):
                #if i >= self.memory_b.len() or self.memory_b.buffer[-(i+1)].id != t_seq.id or i > self.memory_size_b - 1:
                if i >= self.memory_b.len()  or i > self.memory_size_b - 1:
                    break
                self.memory_t.add(self.memory_b.buffer[-(i+1)])

        #store in memory l
        if self.memory_t.len() > 0:
            w_new = max(self.memory_t.buffer, key=(lambda x: x.td_error)).td_error
            w_max = 0
            w_min = 0
            min_id = 0
            for i in range(self.memory_l.len()):
                if w_max < max(self.memory_l.buffer[i].buffer,key=(lambda x: x.td_error)).td_error:
                    w_max = max(self.memory_l.buffer[i].buffer,key=(lambda x: x.td_error)).td_error
                if w_min > min(self.memory_l.buffer[i].buffer,key=(lambda x: x.td_error)).td_error:
                    w_min = min(self.memory_l.buffer[i].buffer,key=(lambda x: x.td_error)).td_error
                    min_id = i
            if w_new * self.tau > w_max:
                self.memory_l.add_copy(self.memory_t,min_id)

        # make virtual sequence
        if self.memory_l.len() > 0:
            self.make_virtual_sequence()

    def store_proposed(self, t_seq,td_error):
        # store in memory b
        if self.memory_b.len() >= self.memory_size:
            del_state = self.memory_b.buffer[0].next_state
            self.memory_state.delete(del_state)
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)
        self.memory_state.add(t_seq.next_state,t_seq)

    def store_proposed2(self, t_seq, td_error):
        # store in memory b
        if self.memory_b.len() >= self.memory_size:
            del_state = self.memory_b.buffer[0].next_state
            self.memory_state.delete(del_state)
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)
        self.memory_state.add(t_seq.next_state, t_seq)

        # store in memory t
        if t_seq.reward > self.threshold_reward:
            for i in range(self.memory_size_t):
                self.memory_t.add(t_seq)

        # make virtual sequence
        if self.memory_t.len() > 0:
            self.make_virtual_sequence2()

    def learning(self,t_seq):
        self.q_func_t[t_seq.state[0], t_seq.state[1], t_seq.action] = \
                        self.q_func_t[t_seq.state[0], t_seq.state[1], t_seq.action] + self.alpha \
                        * (t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]])
                           - self.q_func_t[t_seq.state[0], t_seq.state[1], t_seq.action])

    def learning_er(self):
        for i in range(self.batch_size_per):
            r = random.randint(0, self.memory_b.len() - 1)
            self.learning(self.memory_b.buffer[r])

    def learning_per(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size_per)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        batch_memory = Memory(max_size=self.batch_size_per)
        idx_memory = Memory(max_size=self.batch_size_per)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(self.memory_td.buffer[idx]) ** self.per_alpha + 0.0001
                idx += 1

            batch_memory.add(self.memory_b.buffer[idx])
            idx_memory.add(idx)
        self.goal_num = self.goal_num + batch_memory.len()
        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[
                                                                  state_b[0], state_b[1], action_b] + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
        batch_memory.clear()
        idx_memory.clear()

    def learning_ts(self):
        # あとはこのバッチで学習する

        for j,seq in enumerate(self.memory_l.buffer):
            for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(seq.buffer):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
                td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
                self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[state_b[0], state_b[1], action_b] + self.alpha * td
                self.memory_l.buffer[j].buffer[i] = self.memory_l.buffer[j].buffer[i]._replace(td_error=abs(td))

        for seq in self.memory_v.buffer:
            for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(seq.buffer):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
                td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
                self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[state_b[0], state_b[1], action_b] + self.alpha * td

    def learning_noise_vs(self, var):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        batch_memory = Memory(max_size=self.batch_size)
        idx_memory = Memory(max_size=self.batch_size)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(self.memory_td.buffer[idx]) + 0.0001
                idx += 1

            batch_memory.add(self.memory_b.buffer[idx])
            idx_memory.add(idx)
        #virtual sequence作成
        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(batch_memory.buffer):
            while(1):
                state_b_noise = [int(round(i + np.random.normal(0,var))) for i in state_b]
                if state_b_noise[0] < 0 or state_b_noise[0] > self.shapexy-1 or state_b_noise[1] < 0 or state_b_noise[1] > self.shapexy-1:
                    continue
                if self.world[state_b_noise[0], state_b_noise[1]] == 1:
                    continue
                next_state_b_noise = [int(round(i + np.random.normal(0, var))) for i in next_state_b]
                if next_state_b_noise[0] < 0 or next_state_b_noise[0] > self.shapexy-1 or next_state_b_noise[1] < 0 or next_state_b_noise[1] > self.shapexy-1:
                    continue
                if self.world[next_state_b_noise[0], next_state_b_noise[1]] == 1:
                    continue
                break
            reward_b_noise = reward_b
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b_noise[0], next_state_b_noise[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b_noise + self.gamma * next_action_q - self.q_func_t[state_b_noise[0], state_b_noise[1], action_b]
            self.q_func_t[state_b_noise[0], state_b_noise[1], action_b] = self.q_func_t[state_b[0], state_b_noise[1], action_b] + self.alpha * td * var

        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[state_b[0], state_b[1], action_b] + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
        batch_memory.clear()
        idx_memory.clear()

    def learning_proposed(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        batch_memory = Memory(max_size=self.batch_size)
        idx_memory = Memory(max_size=self.batch_size)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(self.memory_td.buffer[idx]) ** self.per_alpha + 0.0001
                idx += 1

            batch_memory.add(self.memory_b.buffer[idx])
            idx_memory.add(idx)

        self.make_virtual_sequence_proposed(batch_memory)
        # あとはこのバッチで学習する

        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[
                                                                  state_b[0], state_b[1], action_b] + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)

        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(self.memory_v.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[
                                                                  state_b[0], state_b[1], action_b] + self.alpha * td
        batch_memory.clear()
        idx_memory.clear()

    def learning_proposed2(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        batch_memory = Memory(max_size=self.batch_size)
        idx_memory = Memory(max_size=self.batch_size)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(self.memory_td.buffer[idx]) ** self.per_alpha + 0.0001
                idx += 1

            batch_memory.add(self.memory_b.buffer[idx])
            idx_memory.add(idx)

        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[
                                                                  state_b[0], state_b[1], action_b] + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)

        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(self.memory_v.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[
                                                                  state_b[0], state_b[1], action_b] + self.alpha * td
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
                tmp_sum_absolute_TDerror += abs(abstd) ** self.per_alpha + 0.0001
                idx += 1
            id = idx
            '''
            while abstd >= 0 and id <= self.memory_b.len()-1:
                batch_memory.add(self.memory_b.buffer[id])
                idx_memory.add(id)
                abstd -= 10
                id += 1
            '''
            while 1:
                batch_memory.add(self.memory_b.buffer[id])
                idx_memory.add(id)
                abstd -= 1.4
                id -= 1
                if abstd < 0 or id < 0:
                    break
        self.goal_num = self.goal_num + batch_memory.len()
        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, _, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[
                                                                  state_b[0], state_b[1], action_b] + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
        batch_memory.clear()
        idx_memory.clear()

    def episode(self):
        self.sequence_id = 0

        if self.mode == 3:
            self.memory_state.clear()

        for i_episode in tqdm(range(self.num_episodes)):
            # Initialize the Grid_World and state
            state, reward, done = self.env.reset_env()

            sum_reward = 0
            for t in range(50):
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                sum_reward = sum_reward + reward
                #if done:
                    #self.goal_num += 1
                if self.mode == 3:
                    self.memory_state.add(state, self.sequence_id)

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
                elif self.mode ==3: #eruts
                    self.store_ts(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_ts()
                elif self.mode == 4: # noise vs
                    self.store_per(t_seq, td_error)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_noise_vs(var=500/(500+i_episode))
                elif self.mode == 5: # proposed
                    self.store_proposed(t_seq,td_error)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_proposed()
                elif self.mode == 6: # proposed
                    self.store_proposed2(t_seq,td_error)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_proposed2()
                elif self.mode == 7: # proposed
                    self.store_proposed3(t_seq)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_proposed3()
                else:
                    print("mode error")
                    sys.exit()
                # Move to the next state
                state = next_state
                self.q_func = copy.deepcopy(self.q_func_t)

            if self.mode in [2,4,5,6,7]:  # PER
                self.memory_td.update_TDerror(self.memory_b, self.gamma, self.q_func, self.q_func_t)

            self.sum_reward_mem[i_episode] = self.sum_reward_mem[i_episode] + sum_reward

            self.sequence_id += 1
            #self.plot_q(self.q_func)
            #self.anneal_epsilon()
        self.plot_q(self.q_func)

    def experiment(self):
        self.sum_reward_mem = np.zeros(self.num_episodes)
        for ep_num in range(self.num_ex):
            self.episode()
            self.reset()
        self.plot_result(self.sum_reward_mem / self.num_ex)
    def reset(self):
        self.q_func_t = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.q_func = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.memory_b.clear()
        self.memory_td.clear()
        if self.mode == 3:
            self.memory_l.clear()
            self.memory_state.clear()
        if self.mode == 6:
            self.memory_t.clear()


    def make_virtual_sequence(self):
        self.memory_v.clear()
        for seq in self.memory_l.buffer:
            self.memory_vs.add(seq.buffer[0])
            for i,tra in enumerate(seq.buffer):
                if tra.reward > self.threshold_reward:
                    continue
                #if i < 5:
                    #continue
                self.memory_vs.add(tra)
                id_list = copy.deepcopy(self.memory_state.retrieve(tra.state))
                id_list.remove(tra.id)
                if len(id_list) == 0:
                    continue
                material_id = random.choice(id_list)
                for seq_b in self.memory_b.buffer:
                    if seq_b.id == material_id:
                        if np.allclose(tra.state, seq_b.state):
                            break
                        self.memory_vs2.add(seq_b)
                break
            if self.memory_vs2.len() > 0:
                self.memory_vs2.reverse()
                for i in self.memory_vs2.buffer:
                    self.memory_vs.add(i)
                self.memory_v.add_copy(self.memory_vs, 0)
            else:
                self.memory_v.clear()
            self.memory_vs.clear()
            self.memory_vs2.clear()

    def make_virtual_sequence2(self):
        self.memory_v.clear()
        for seq in self.memory_t.buffer:
            # self.memory_v.add(seq)
            tmp_seq = seq
            for i in range(self.lenv):
                seq_list = self.memory_state.retrieve(tmp_seq.next_state)
                if len(seq_list) == 0:
                    break
                tmp_seq = random.choice(seq_list)
                self.memory_v.add(tmp_seq)

    def make_virtual_sequence_proposed(self, batch_memory):
        self.memory_v.clear()
        for seq in batch_memory.buffer:
            #self.memory_v.add(seq)
            tmp_seq = seq
            for i in range(self.lenv):
                seq_list = self.memory_state.retrieve(tmp_seq.next_state)
                if len(seq_list) == 0:
                    break
                tmp_seq = random.choice(seq_list)
                self.memory_v.add(tmp_seq)

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
        print(self.goal_num)
        self.goal_num = 0
    def plot_result(self, result):
        plt.plot(result, label=self.mode)




if __name__ == '__main__':
    for i in [1,2,7]:
        q_learning = Q_Learning(mode=i)
        q_learning.experiment()
    plt.legend()
    plt.show()