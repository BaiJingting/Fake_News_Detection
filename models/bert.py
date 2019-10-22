# -*- coding: utf-8 -*-

import os
import sys
import codecs
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras_radam import RAdam
from keras_bert import load_trained_model_from_checkpoint, get_custom_objects
from keras.callbacks import ModelCheckpoint, EarlyStopping

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

from data_generator import DataGenerator, OurTokenizer, seq_padding
import process

model_name = 'bert'  # 'bert' / 'roeberta'

CONFIG_PATH = os.path.join(ROOT_PATH, 'model_files', model_name, 'initial_model/config.json')
CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'model_files', model_name, 'initial_model/model.ckpt')
DICT_PATH = os.path.join(ROOT_PATH, 'model_files', model_name, 'initial_model/vocab.txt')

CONFIG = {
    'max_len': 300,
    'batch_size': 8,
    'epochs': 3,
    'use_multiprocessing': True,
    'model_dir': os.path.join(ROOT_PATH, 'model_files/bert'),
    'trainable_layers': 26
}


class BertClassify:
    def __init__(self, initial_model=True, model_path=os.path.join(CONFIG['model_dir'], 'bert.h5')):
        self.initial_model = initial_model
        self.model_path = model_path
        if initial_model:
            self.bert_model = load_trained_model_from_checkpoint(config_file=CONFIG_PATH,
                                                                 checkpoint_file=CHECKPOINT_PATH)
        else:
            self.load(model_path)

        # # 资源不允许的情况下只训练部分层的参数
        # for layer in self.bert_model.layers[: -CONFIG['trainable_layers']]:
        #     layer.trainable = False
        # # 资源允许的话全部训练
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
        train_D = DataGenerator(train_data, self.tokenizer, CONFIG['batch_size'], CONFIG['max_len'])
        valid_D = DataGenerator(valid_data, self.tokenizer, CONFIG['batch_size'], CONFIG['max_len'])

        save = ModelCheckpoint(
            os.path.join(self.model_path),
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
        if self.initial_model:
            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))

            x_in = self.bert_model([x1_in, x2_in])
            x_in = Lambda(lambda x: x[:, 0])(x_in)
            p = Dense(1, activation='sigmoid')(x_in)
            self.model = Model([x1_in, x2_in], p)
        else:
            self.model = self.bert_model

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=RAdam(1e-5),  # 用足够小的学习率
            metrics=['accuracy', process.get_precision, process.get_recall, process.get_f1]
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
        X1 = []
        X2 = []
        for s in test_data:
            x1, x2 = self.tokenizer.encode(first=s[:CONFIG['max_len']])
            X1.append(x1)
            X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        predict_results = self.model.predict([X1, X2])
        return predict_results

    def load(self, model_path):
        """
        load the pre-trained model
        """
        try:
            self.bert_model = load_model(str(model_path), custom_objects=get_custom_objects(), compile=False)
        except Exception as ex:
            print('load error')
        return self
