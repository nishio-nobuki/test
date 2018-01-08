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

# [※p3] Memoryクラスを継承した、TD誤差を格納するクラスです
class MemoryTDerror(Memory):
    def __init__(self, max_size=1000):
        super().__init__(max_size)

    # add, sample, len は継承されているので定義不要

    # TD誤差をすべて更新
    def update_TDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            (state, action, reward, next_state, _) = memory.buffer[i]  # 最新の状態データを取り出す
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action_q = np.argmax(mainQN[next_state[0], next_state[1], :])  # 最大の報酬を返す行動を選択する
            td = reward + gamma * next_action_q - targetQN[state[0], state[1], action]
            self.buffer[i] = td

    # TD誤差の絶対値和を取得
    def get_sum_absolute_TDerror(self):
        sum_absolute_TDerror = 0
        for i in range(0, (self.len() - 1)):
            sum_absolute_TDerror += abs(self.buffer[i]) + 0.0001  # 最新の状態データを取り出す

        return sum_absolute_TDerror
