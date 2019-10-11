# -*- coding: utf-8 -*-

import os
import re
import jieba
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from keras import backend as K

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def get_char_type(c):
    if ord(c) in range(48, 58):
        return 'digit'
    elif ord(c) in range(65, 91) or ord(c) in range(97, 123):
        return 'en'
    else:
        return 'other'


def get_char_seg(text):
    """
    对文本切分成单字，保留英文单词及数字
    :param text:
    :return:
    """
    ret = []
    item = ''
    label = 'other'  # 'en'英文字符; 'digit'数字; 'other'其他
    for char in text:
        char_type = get_char_type(char)
        if char_type == 'other':
            ret.append(item)
            item = char
        else:
            if label != char_type:
                ret.append(item)
                item = char
            else:
                item += char
        label = char_type
    ret.append(item)
    return ret[1:]


def get_seg(text):
    """
    分词
    :param text:
    :return:
    """
    return list(jieba.cut(text))


def split_train_test(data, X_name, y_name, train_size=0.85, test_size=None):
    """
    对数据切分成训练集和测试集
    :param data:
    :param X_name:
    :param y_name:
    :param train_size:
    :return:
    """
    train_data = []
    test_data = []
    if (not train_size) and test_size:
        train_size = 1 - test_size
    for i in range(data.shape[0]):
        if i % 100 < train_size * 100:
            train_data.append([data.loc[i][X_name], data.loc[i][y_name]])
        else:
            test_data.append([data.loc[i][X_name], data.loc[i][y_name]])
    return np.array(train_data), np.array(test_data)


def pretreatment(text):
    """
    replace url, email and phone num by specific strings
    """
    text = re.sub(r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,]|"
                  r"(?:%[0-9a-fA-F][0-9a-fA-F]))+)|([a-zA-Z]+.\w+\.+[a-zA-Z0-9\/_]+)", '链接', text)
    text = re.sub(r"1\d{10}", '手机号', text)
    text = re.sub(r"\d{3,4}-\d{8}", '座机', text)
    text = re.sub(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", '邮箱', text)
    text = re.sub(r"(//@).+?[:：]", ' ', text)
    text = re.sub(r"&.{1,6};", ' ', text)
    text = re.sub('\r', ' ', text)
    text = get_char_seg(text)
    return text


def submit_data(model, seg_fun):
    input_path = os.path.join(ROOT_PATH, 'data/test_stage1.csv')
    out_path = os.path.join(ROOT_PATH, 'data/submit.csv')
    data = pd.read_csv(input_path)
    text_seg = [' '.join(item) for item in data.text.apply(lambda x: seg_fun(x))]
    labels = model.predict(text_seg)
    data_out = pd.DataFrame({'id': data.id, 'label': labels})
    data_out.to_csv(out_path, index=False, encoding='utf-8')


def get_precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def get_recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def get_f1(y_true, y_pred):
    """
    f1 metric
    :param y_true:
    :param y_pred:
    :return:
    """
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == "__main__":
    text = '里面有身份证刘耀峰，现金6000左右，卡6000左右'
    print('\n'.join(get_char_seg(text)))
