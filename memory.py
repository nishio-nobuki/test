#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf
import copy

#for reshape
def rs12(state):
    return np.reshape(state, [1, 2])

class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, experience):
        self.buffer.append(experience)

    def add_copy(self, experience, id):
        seq = Memory(experience.len())
        for i in experience.buffer:
            seq.add(i)
        if self.len() >= self.max_size:
            self.buffer[id] = seq
        else:
            self.buffer.append(seq)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

    def max(self):
        return max(self.buffer)

    def abs_max(self):
        return max(self.buffer, key=abs)

    def min(self):
        return min(self.buffer)

    def abs_sum(self):
        sum = 0
        for i in range(0, (self.len() - 1)):
            sum += abs(self.buffer[i])  # 最新の状態データを取り出す

        return sum

    def sum(self):
        sum = 0
        for i in range(0, (self.len() - 1)):
            sum += self.buffer[i] # 最新の状態データを取り出す

        return sum

    def clear(self):
        self.buffer.clear()

    def assign(self,i,num):
        self.buffer[i] = num

    def reverse(self):
        self.buffer.reverse()

    def pop(self):
        self.buffer.popleft()

    def clear(self):
        self.buffer.clear()

# [※p3] Memoryクラスを継承した、TD誤差を格納するクラスです
class MemoryTDerror(Memory):
    def __init__(self, max_size=1000, alpha=1.0):
        super().__init__(max_size)
        self.alpha = alpha

    # add, sample, len は継承されているので定義不要

    # TD誤差をすべて更新
    def update_TDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            (state, action, reward, next_state, _, _) = memory.buffer[i]  # 最新の状態データを取り出す
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）

            next_action = np.argmax(mainQN[next_state[0], next_state[1], :])  # 最大の報酬を返す行動を選択する
            td = reward + gamma * targetQN[next_state[0], next_state[1], next_action] - mainQN[state[0], state[1], action]
            self.buffer[i] = td

    def update_DeepTDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            (state, action, reward, next_state, _, _) = memory.buffer[i]  # 最新の状態データを取り出す
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）

            next_action = np.argmax(mainQN.model.predict(rs12(next_state))[0]) # 最大の報酬を返す行動を選択する
            td = reward + gamma * targetQN.model.predict(rs12(next_state))[0][next_action] - mainQN.model.predict(rs12(state))[0][action]
            self.buffer[i] = td

    # TD誤差の絶対値和を取得
    def get_sum_absolute_TDerror(self):
        sum_absolute_TDerror = 0
        for i in range(0, (self.len() - 1)):
            sum_absolute_TDerror += abs(self.buffer[i]) ** self.alpha + 0.0001  # 最新の状態データを取り出す

        return sum_absolute_TDerror

    def anneal_per_alpha_mem(self,num):
        self.alpha = num

class StateMemory():
    def __init__(self, x, y):
        self.state_memory = [[[] for j in range(x)] for i in range(y)]
        self.size_x = x
        self.size_y = y

    def add(self, state, id):
        if id not in self.state_memory[state[0]][state[1]]:
            self.state_memory[state[0]][state[1]].append(id)

    def delete_state(self,state, id):
        self.state_memory[state[0]][state[1]].remove(id)

    def retrieve(self, state):
        return self.state_memory[state[0]][state[1]]

    def clear(self):
        self.state_memory = [[[] for j in range(self.size_x)] for i in range(self.size_y)]

class ProposedMemory():
    def __init__(self, x, y):
        self.state_memory = [[[] for j in range(x)] for i in range(y)]
        self.size_x = x
        self.size_y = y

    def add(self, state, t_seq):
        if id not in self.state_memory[state[0]][state[1]]:
            self.state_memory[state[0]][state[1]].append(t_seq)

    def delete(self,state):
        self.state_memory[state[0]][state[1]].pop(0)

    def retrieve(self, state):
        return self.state_memory[state[0]][state[1]]

    def clear(self):
        self.state_memory = [[[] for j in range(self.size_x)] for i in range(self.size_y)]

class MultiMemory():
    def __init__(self, memory_num, batch_size):
        self.memory_num = memory_num
        self.memory_size = batch_size
        self.batch_memory = []
        self.idx_memory = []
        for i in range(self.memory_num):
            self.batch_memory.append(Memory(max_size=batch_size))
            self.idx_memory.append(Memory(max_size=batch_size))

    def shift_memory(self):
        for i in range(self.memory_num-1):
            self.batch_memory[i] = copy.deepcopy(self.batch_memory[i+1])
            self.idx_memory[i] = copy.deepcopy(self.idx_memory[i + 1])
        self.batch_memory[self.memory_num - 1].clear()
        self.idx_memory[self.memory_num - 1].clear()

    def clear_all_memory(self):
        for i in range(self.memory_num):
            self.batch_memory[i].clear()
            self.idx_memory[i].clear()