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
import matplotlib.pyplot as plt
from tqdm import tqdm
from gridworld import Grid_World
from memory import Memory, MemoryTDerror, StateMemory, ProposedMemory

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
        self.goal = [[1, 1], [12, 12]]  #2つまで
        self.shapexy = 15  #gridworldのサイズ
        self.count = 0
        self.gamma = 0.9
        self.alpha = 0.1
        self.per_alpha_start = 0.6
        self.per_alpha = self.per_alpha_start
        self.reward_def = [-10, -100, 100, 10]  # 何もない時、壁に当たった時、ゴール1、ゴール2
        self.num_episodes = 2000 #500
        self.num_ex = 2 #50
        self.eps_start = 0.1
        self.eps = self.eps_start + 0.01
        self.batch_size = 5
        self.batch_size_per = 8
        self.memory_size = 1000
        self.beta_start = 1.0
        self.beta = self.beta_start
        self.atb_p = 0
        self.atb_len = 1 #1

        self.env = Grid_World(self.world, self.goal, self.reward_def, self.shapexy)
        self.sum_reward_mem = np.zeros(self.num_episodes)
        self.mem_avr_td = np.zeros(self.num_episodes)
        self.mem_max_td = np.zeros(self.num_episodes)
        self.memory_b = Memory(max_size=self.memory_size)
        self.memory_td = MemoryTDerror(max_size=self.memory_size,alpha=self.per_alpha)
        self.threshold_reward = 10
        self.q_func_t = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.q_func = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.sequence_id = 0
        self.update_num = 0
        self.ave_update_num = 0
        self.i_episode = 0

        self.t = 0

        # if agent quits the episode and restart when it arrives at the goal: 1
        self.goal_and_end = 0

        #if you adapt the  batch size of proposed method to all other method : 1
        self.adapt_batch_size = 1
        self.update_num_list = np.zeros(self.num_episodes * 50)
        self.un = self.update_num_list


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
        td_error = t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]]) \
                   - self.q_func[t_seq.next_state[0], t_seq.next_state[1], t_seq.action]
        t_seq = t_seq._replace(td_error=abs(td_error))
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def learning(self,t_seq):
        self.q_func_t[t_seq.state[0], t_seq.state[1], t_seq.action] = \
                        self.q_func_t[t_seq.state[0], t_seq.state[1], t_seq.action] + self.alpha \
                        * (t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]])
                           - self.q_func_t[t_seq.state[0], t_seq.state[1], t_seq.action])

    def learning_er(self):
        for i in range(self.batch_size_per):
            r = random.randint(0, self.memory_b.len() - 1)
            self.learning(self.memory_b.buffer[r])
        self.update_num = self.update_num + self.batch_size_per

    def learning_per(self):
        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = self.memory_td.get_sum_absolute_TDerror()
        if self.adapt_batch_size == 1:
            generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.un[self.i_episode * 50 + self.t])
        else:
            generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, self.batch_size_per)

        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        #batch_memory = Memory(max_size=self.batch_size_per)
        #idx_memory = Memory(max_size=self.batch_size_per)
        batch_memory = Memory(max_size=self.un[self.i_episode*50+self.t])
        idx_memory = Memory(max_size=self.un[self.i_episode*50+self.t])
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
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[state_b[0], state_b[1], action_b] \
                                                              + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
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
            '''
            while 1:
                abstd -= self.atb_len
                id -= 1
                if abstd < 0 or id < 0 or self.memory_b.buffer[idx].id != self.memory_b.buffer[id].id:
                    break
                batch_memory.add(self.memory_b.buffer[id])
                idx_memory.add(id)

        self.update_num = self.update_num + batch_memory.len()
        self.update_num_list[self.i_episode*50+self.t] += batch_memory.len()
        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, td_error, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[state_b[0], state_b[1], action_b] \
                                                              + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
        batch_memory.clear()
        idx_memory.clear()

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
                else:
                    print("mode error")
                    sys.exit()
                # Move to the next state
                state = next_state
                self.q_func = copy.deepcopy(self.q_func_t)

            if self.mode in [2,4,5,6,7,8,9]:  # PER
                self.memory_td.update_TDerror(self.memory_b, self.gamma, self.q_func, self.q_func_t)

            self.sum_reward_mem[self.i_episode] = self.sum_reward_mem[self.i_episode] + sum_reward
            self.mem_max_td[self.i_episode] = self.memory_td.max()
            self.mem_avr_td[self.i_episode] = (self.memory_td.get_sum_absolute_TDerror() / self.memory_size)

            self.sequence_id += 1
            #self.plot_q(self.q_func)
            #self.anneal_epsilon()
            if self.mode in [2,7]:
                self.anneal_per_alpha()
            #self.anneal_beta()
            if self.mode in [7]:
                self.anneal_atb_len(self.memory_td.max())

            #mode break

            if self.goal_and_end == 1:
                if sum_reward > 1250:
                    self.count += 1
                    if self.count >= 10:
                        break
                else:
                    self.count = 0


        self.plot_q(self.q_func)
        self.ave_update_num += self.update_num / self.num_ex
        self.update_num = 0

    def experiment(self):
        elapsed_time = []
        for self.mode in self.modeall:
            self.sum_reward_mem = np.zeros(self.num_episodes)
            for ep_num in tqdm(range(self.num_ex)):
                start = time.time()
                self.episode()
                self.reset()
                elapsed_time.append(time.time() - start)
                print(elapsed_time[-1])
            print(self.ave_update_num)

            #return [int(round(x/self.num_ex)) for x in self.update_num_list]
            self.un = [int(round(x / self.num_ex)) for x in self.update_num_list]

            plt.subplot(2, 2, 1)
            self.plot_result(self.sum_reward_mem / self.num_ex)
            #plt.legend()
            plt.title(r"sum of reward")

        plt.subplot(2, 2, 2)
        self.plot_result(self.un)
        #plt.legend()
        plt.title(r"update num")

        plt.subplot(2, 2, 3)
        self.plot_result(self.mem_avr_td)
        #plt.legend()
        plt.title(r"max")

        plt.subplot(2, 2, 4)
        self.plot_result(self.mem_max_td)
        #plt.legend()
        plt.title(r"avr")
        plt.show()

        print(sum(elapsed_time)/self.num_ex)

    def reset(self):
        self.q_func_t = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.q_func = np.zeros(shape=(self.shapexy, self.shapexy, 5))
        self.memory_b.clear()
        self.memory_td.clear()
        if self.mode in [7]:
            self.atb_p = 0
            self.atb_len = 1
        if self.mode in [2,7]:
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

    def plot_result(self, result):
        if self.mode == 0:  # normal
            label = 'Q Learning'
        elif self.mode == 1:  # experience replay
            label = 'Experience Replay(uniform)'
        elif self.mode == 2:  # PER
            label = 'Prioritized Experience Replay'
        elif self.mode == 7:  # proposed
            label = 'Proposal Method'
        else:
            print("mode error")
            sys.exit()

        plt.plot(result, label=label)
        print(self.ave_update_num)



if __name__ == '__main__':
    '''
    modeについて
    1:normal
    2:using experience replay
    3:per
    7:proposed method
    '''
    modeall = [7]
    q_learning = Q_Learning(mode=modeall)
    q_learning.experiment()
    #print(tmp)





    #plt.legend()
    #plt.show()