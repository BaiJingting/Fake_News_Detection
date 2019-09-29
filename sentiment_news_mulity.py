#! -*- coding:utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from random import choice

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

from keras.utils import multi_gpu_model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs

max_len = 128
config_path = os.path.join(ROOT_PATH, 'model_files/bert/chinese_L-12_H-768_A-12/bert_config.json')
checkpoint_path = os.path.join(ROOT_PATH, 'model_files/bert/chinese_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(ROOT_PATH, 'model_files/bert/chinese_L-12_H-768_A-12/vocab.txt')

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

neg = pd.read_csv(os.path.join(ROOT_PATH, 'data/train.csv')).values.tolist()

data = []

for d in neg:
    if d[1] is not None and d[2] is not None:
        data.append((d[1].decode('utf-8'), d[2]))

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
test_file = pd.read_csv(os.path.join(ROOT_PATH, 'data/test_stage1.csv')).values.tolist()
test_data = []
for i in test_file:
    if i[1] is not None:
        test_data.append(i[1].decode('utf-8'))


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:max_len]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
import keras_radam as Radam

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)
train_D = data_generator(train_data)
valid_D = data_generator(valid_data)
test_D = data_generator(test_data)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Radam.RAdam(1e-5),  # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()
model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=10,
    epochs=1,
    validation_data=valid_D.__iter__(),
    use_multiprocessing=True,
    validation_steps=len(valid_D)
)
model.save_weights(os.path.join(ROOT_PATH, 'model_files/bert/best_model.weights'))

X1 = []
X2 = []
for s in test_data:
    x1, x2 = tokenizer.encode(first=s[:max_len])
    X1.append(x1)
    X2.append(x2)
X1 = seq_padding(X1)
X2 = seq_padding(X2)
predict_results = model.predict([X1, X2])
with open(os.path.join(ROOT_PATH, 'data/bert/predict.txt'), 'w') as f:
    f.write(predict_results)
