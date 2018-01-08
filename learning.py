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
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
from tqdm import tqdm
from gridworld import Grid_World
from memory import Memory, MemoryTDerror

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'td_error'))

class Q_Learning():
    def __init__(self, mode):
        self.mode = mode
        self.gamma = 0.9
        self.alpha = 0.3
        self.goal = [[28, 27], [1, 1]]
        self.reward_def = [-10, -100, 100, 100]  # 何もない時、壁に当たった時、ゴール1、ゴール2
        self.num_episodes = 1000
        self.num_ex = 1
        self.eps = 0.1
        self.tau = 1.5
        self.batch_size = 15
        self.memory_size = 1000
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
        self.env = Grid_World(self.world, self.goal, self.reward_def)
        self.sum_reward_mem = np.zeros(self.num_episodes)
        self.memory_b = Memory(max_size=self.memory_size)
        self.memory_td = MemoryTDerror(max_size=self.memory_size)
        self.threshold_reward = 1
        self.q_func_t = np.zeros(shape=(30, 30, 5))
        self.q_func = self.q_func_t

    def select_action(self, state):
        sample = random.random()
        if sample > self.eps:
            q_value = self.q_func[state[0], state[1]]
            action = np.argmax(q_value)
        else:
            action = random.randint(0, 4)
        return action

    def store(self, t_seq):
        # calc TD-error
        td_error = t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0],t_seq.next_state[1]]) \
                   - self.q_func[t_seq.next_state[0],t_seq.next_state[1],t_seq.action]
        t_seq.td_error = abs(td_error)

        #store in memory b
        if len(self.memory_b) > self.memory_size:
            self.memory_b.pop(0)
        self.memory_b.append(t_seq)

        #store in memory t
        self.memory_t = []
        if t_seq.reward > self.threshold_reward:
            for i in range(self.size_of_memt):
                self.memory_t[i] = self.memory_b[-(i+1)]
                if self.memory_t[i].state == self.start or i > self.memory_size - 2:
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

    def store_er(self, t_seq):
        # store in memory b
        self.memory_b.add(t_seq)

    def store_per(self, t_seq, td_error):
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def store_mu(self, t_seq, td_error):
        # store in memory b
        self.memory_b.add(t_seq)
        self.memory_td.add(td_error)

    def store_ts(self, t_seq):
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

    def learning(self,t_seq):
        self.q_func[t_seq.state[0], t_seq.state[1], t_seq.action] = \
                        self.q_func[t_seq.state[0], t_seq.state[1], t_seq.action] + self.alpha \
                        * (t_seq.reward + self.gamma * max(self.q_func[t_seq.next_state[0], t_seq.next_state[1]])
                           - self.q_func[t_seq.state[0], t_seq.state[1], t_seq.action])

    def learning_er(self):
        for i in range(self.batch_size):
            r = random.randint(0, self.memory_b.len() - 1)
            self.learning(self.memory_b.buffer[r])

    def learning_per(self):
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
        # あとはこのバッチで学習する
        for i, (state_b, action_b, reward_b, next_state_b, _) in enumerate(batch_memory.buffer):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.max(self.q_func[next_state_b[0], next_state_b[1], :])  # 最大の報酬を返す行動を選択する
            td = reward_b + self.gamma * next_action_q - self.q_func_t[state_b[0], state_b[1], action_b]
            self.q_func_t[state_b[0], state_b[1], action_b] = self.q_func_t[
                                                                  state_b[0], state_b[1], action_b] + self.alpha * td
            self.memory_td.assign(idx_memory.buffer[i], td)
        batch_memory.clear()
        idx_memory.clear()

    def episode(self):
        for i_episode in range(self.num_episodes):
            # Initialize the Grid_World and state
            state, reward, done = self.env.reset_env()
            sum_reward = 0
            for t in range(100):
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                sum_reward = sum_reward + reward

                if self.memory_td.len() == 0:
                    td_error = 0
                else:
                    td_error = self.memory_td.max()

                #store the sequence
                t_seq = Transition(state=state,
                                   action=action,
                                   reward=reward,
                                   next_state=next_state,
                                   td_error=td_error)

                if self.mode == 0:  # normal
                    self.learning(t_seq)
                elif self.mode == 1:  #experience replay
                    self.store_er(t_seq)
                    self.learning_er()
                elif self.mode == 2: #PER
                    self.store_per(t_seq,td_error)
                    if self.memory_b.len() >= self.memory_size:
                        self.learning_per()
                elif self.mode ==3: #proposed
                    print("to do")
                else:
                    print("mode error")
                    sys.exit()
                # Move to the next state
                state = next_state

                self.q_func = self.q_func_t

            self.sum_reward_mem[i_episode] = self.sum_reward_mem[i_episode] + sum_reward
            self.memory_td.update_TDerror(self.memory_b,self.gamma,self.q_func,self.q_func_t)
        #self.plot_q(self.q_func)

    def experiment(self):
        self.sum_reward_mem = np.zeros(self.num_episodes)
        for ep_num in tqdm(range(self.num_ex)):
            self.episode()
            self.reset()
        self.plot_result(self.sum_reward_mem / self.num_ex)

    def reset(self):
        self.q_func_t = np.zeros(shape=(30, 30, 5))
        self.q_func = self.q_func_t
        self.memory_b.clear()
        self.memory_td.clear()

    def make_virtual_sequence(self):
        for seq in self.memory_l:
            for tra in seq:
                for trab in self.memory_b:
                    if tra.state == trab.state:
                        intersec = tra.state

    def plot_q(self, Qfunc):
        print('_______________________________')
        for i in range(30):
            for j in range(30):
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
                sys.stdout.write(symbol)
            print('|')
        print('_______________________________')

    def plot_result(self, result):
        plt.plot(result, label=self.mode)




if __name__ == '__main__':
    for i in [0,1,2]:
        q_learning = Q_Learning(mode=i)
        q_learning.experiment()
    plt.legend()
    plt.show()