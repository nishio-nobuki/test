#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import deque



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

    def clear(self):
        self.buffer.clear()

    def assign(self,i,num):
        self.buffer[i] = num

    def reverse(self):
        self.buffer.reverse()

    def pop(self):
        self.buffer.popleft()

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
            next_action_q = np.argmax(mainQN[next_state[0], next_state[1], :])  # 最大の報酬を返す行動を選択する
            td = reward + gamma * next_action_q - targetQN[state[0], state[1], action]
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