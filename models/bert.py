# -*- coding: utf-8 -*-

import os
import codecs
import tensorflow as tf
from keras.layers import *
from keras.models import Model, load_model
import keras_radam as Radam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects
from keras.callbacks import ModelCheckpoint, EarlyStopping

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_PATH, 'model_files/bert/chinese_L-12_H-768_A-12/bert_config.json')
CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'model_files/bert/chinese_L-12_H-768_A-12/bert_model.ckpt')
DICT_PATH = os.path.join(ROOT_PATH, 'model_files/bert/chinese_L-12_H-768_A-12/vocab.txt')

CONFIG = {
    'max_len': 188,
    'batch_size': 16,
    'epochs': 2,
    'use_multiprocessing': True,
    'model_dir': os.path.join(ROOT_PATH, 'model_files/bert')
}


class DataGenerator:

    def __init__(self, data, tokenizer, batch_size=CONFIG['batch_size']):
        self.data = data
        self.tokenizer = tokenizer
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
                text = d[0][:CONFIG['max_len']]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = self.seq_padding(X1)
                    X2 = self.seq_padding(X2)
                    Y = self.seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

    def seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])


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


class BertClassify:
    def __init__(self, train=True):
        if train:
            self.bert_model = load_trained_model_from_checkpoint(config_file=CONFIG_PATH, checkpoint_file=CHECKPOINT_PATH)
            for l in self.bert_model.layers:
                l.trainable = True
        self.model = None
        self.__initial_token_dict()
        self.tokenizer = OurTokenizer(self.token_dict)

    def __initial_token_dict(self):
        self.token_dict = {}
        with codecs.open(DICT_PATH, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

    def train(self, train_data, valid_data):
        """
        训练
        :param train_data:
        :param valid_data:
        :return:
        """
        train_D = DataGenerator(train_data, self.tokenizer)
        valid_D = DataGenerator(valid_data, self.tokenizer)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x_in = self.bert_model([x1_in, x2_in])
        x_in = Lambda(lambda x: x[:, 0])(x_in)
        p = Dense(1, activation='sigmoid')(x_in)

        save = ModelCheckpoint(
            os.path.join(CONFIG['model_dir'], 'bert.h5'),
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )
        early_stopping = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=3,
            verbose=1,
            mode='auto'
        )
        callbacks = [save, early_stopping]

        self.model = Model([x1_in, x2_in], p)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Radam.RAdam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        self.model.summary()
        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=CONFIG['epochs'],
            callbacks=callbacks,
            validation_data=valid_D.__iter__(),
            use_multiprocessing=CONFIG['use_multiprocessing'],
            validation_steps=len(valid_D)
        )

    def predict(self, test_data):
        """
        预测
        :param test_data:
        :return:
        """
        test_D = DataGenerator(test_data, self.tokenizer)
        predict_results = self.model.predict_generator(test_D, use_multiprocessing=True, steps=100, verbose=1)
        labels = (np.argmax(predict_results, axis=1))
        return predict_results, labels

    def load(self, model_dir):
        """
        load the pre-trained model
        """
        model_path = os.path.join(model_dir, 'bert.h5')
        try:
            graph = tf.Graph()
            with graph.as_default():
                session = tf.Session()
                with session.as_default():
                    self.reply = load_model(
                        str(model_path),
                        custom_objects=get_custom_objects(),
                        compile=False
                        )
                    with open(os.path.join(model_dir, 'label_map_bert.txt'), 'r') as f:
                        self.label_map = eval(f.read())
                    self.graph = graph
                    self.session = session
        except Exception as ex:
            print('load error')
        return self
