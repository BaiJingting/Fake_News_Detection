# -*- coding: utf-8 -*-

import multiprocessing
from fasttext import FastText

CONFIG = {
    'lr': 0.01,
    'dim': 100,
    'ws': 10,
    'epoch': 10000,
    'minCount': 5,
    'minCountLabel': 0,
    'minn': 2,
    'maxn': 6,
    'neg': 5,
    'wordNgrams': 2,
    'loss': "softmax",
    'bucket': 2000000,
    'thread': multiprocessing.cpu_count() - 1,
    'lrUpdateRate': 100,
    't': 1e-4,
    'label': "__label__",
    'verbose': 2,
    'pretrainedVectors': "",
}


class FastTextClassify:

    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path

    def save_train_file(self, train_X, train_y, file_path):
        """
        将训练数据保存成特定格式的文件
        :param train_X:
        :param train_y:
        :return:
        """
        assert len(train_X) == len(train_y)
        with open(file_path, 'w') as f:
            for i in range(len(train_X)):
                f.write(' '.join(train_X[i]) + '\t' + '__label__' + str(train_y[i]) + '\n')

    def save_test_file(self, test_X, file_path):
        """
        将测试数据保存成特定格式的文件
        :param test_X:
        :param file_path:
        :return:
        """
        with open(file_path, 'w') as f:
            for i in range(len(test_X)):
                f.write(' '.join(test_X[i]) + '\n')

    def train(self, file_path):
        """
        训练
        :return:
        """
        self.model = FastText.train_supervised(input=file_path, **CONFIG)

    def test(self, file_path):
        """
        测试
        :param file_path:
        :return:
        """
        result = self.model.test(file_path)
        print(result)

    def predict(self, data):
        """
        预测
        :param data:
        :return:
        """
        labels = self.model.predict(data)
        return [int(item[0].lstrip('__label__')) for item in labels[0]]
