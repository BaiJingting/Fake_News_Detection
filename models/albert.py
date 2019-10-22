# -*- coding: utf-8 -*-
import os
import sys
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras_bert import get_custom_objects
from bert4keras.bert import load_pretrained_model
from bert4keras.train import PiecewiseLinearLearningRate
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.callbacks import ModelCheckpoint, EarlyStopping

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
import process

model_name = 'albert'

CONFIG_PATH = os.path.join(ROOT_PATH, 'model_files', model_name, 'initial_model/config.json')
CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'model_files', model_name, 'initial_model/model.ckpt')
DICT_PATH = os.path.join(ROOT_PATH, 'model_files', model_name, 'initial_model/vocab.txt')

from data_generator import DataGenerator, seq_padding

CONFIG = {
    'max_len': 256,
    'batch_size': 8,
    'epochs': 3,
    'use_multiprocessing': True,
    'model_dir': os.path.join(ROOT_PATH, 'model_files/albert'),
    'trainable_layers': 26
}


class AlbertClassify:
    def __init__(self, initial_model=True, model_path=os.path.join(CONFIG['model_dir'], 'albert.h5')):
        self.initial_model = initial_model
        token_dict = load_vocab(DICT_PATH)
        self.tokenizer = SimpleTokenizer(token_dict)
        self.model_path = model_path
        if initial_model:
            self.albert_model = load_pretrained_model(
                CONFIG_PATH,
                CHECKPOINT_PATH,
                # keep_words=keep_words,
                albert=True
            )
        else:
            self.load(model_path)

        for l in self.albert_model.layers:
            l.trainable = True

    def train(self, train_data, valid_data):
        train_D = DataGenerator(train_data, self.tokenizer, CONFIG['batch_size'], CONFIG['max_len'])
        valid_D = DataGenerator(valid_data, self.tokenizer, CONFIG['batch_size'], CONFIG['max_len'])

        output = Lambda(lambda x: x[:, 0])(self.albert_model.output)
        output = Dense(1, activation='sigmoid')(output)
        self.model = Model(self.albert_model.input, output)

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

            x_in = self.albert_model([x1_in, x2_in])
            x_in = Lambda(lambda x: x[:, 0])(x_in)
            p = Dense(1, activation='sigmoid')(x_in)
            self.model = Model([x1_in, x2_in], p)
        else:
            self.model = self.albert_model

        self.model.compile(
            loss='binary_crossentropy',
            # optimizer=RAdam(1e-5),  # 用足够小的学习率
            optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
            metrics=['accuracy', process.get_precision, process.get_recall, process.get_f1]
        )
        self.model.summary()

        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=CONFIG['epochs'],
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=callbacks,
            use_multiprocessing=CONFIG['use_multiprocessing'],
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
            self.albert_model = load_model(str(model_path), custom_objects=get_custom_objects(), compile=False)
        except Exception as ex:
            print('load error')
        return self
