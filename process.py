# -*- coding: utf-8 -*-

import os
import re
import jieba
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def get_char_seg(text):
    """
    对文本切分成单字，保留英文单词及数字
    :param text:
    :return:
    """
    return [item for item in text]


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


if __name__ == "__main__":
    text = '15600926082装病我们早就说过了啊，崔老师不会这么忠实&gt;剧本吧？毫无创意//@henryhunter:据说是装病。' \
           '//@连山居士飞扬:应该不会吧？毕竟是有病的人。//@henryhunter:网传崔大炮被抓。//@司马3忌:崔老师三天没敲锣了，城管来了？'
    print(pretreatment(text))
