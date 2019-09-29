# -*- coding: utf-8 -*-

import os
import jieba
import pandas as pd
# import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def analysis(data):
    """
    对数据做描述性统计分析，了解概况
    :return:
    """
    # # 了解样本整体情况
    # 是否有重复样本
    print(data.text.nunique(), data.shape[0])
    # 查看重复样本文本情况
    print(data.groupby(by='text').filter(lambda x: x.label.count() > 1).text)
    # 是否存在文本相同、标注不同的情况
    print(data.groupby(by=['text', 'label']).count().shape)
    # 正负例是否均匀
    print(data.groupby(by='label').count())
    # 缺失值处理（这里没有缺失值）
    print(data[data.isna().values])

    # # 了解文本情况
    length = data.text.apply(lambda x: len(x.decode('utf-8')))
    print(length.describe())
    # 5%、10%、90%、95%分位数分别为：28、41、152、188
    print(length.quantile(0.05), length.quantile(0.1), length.quantile(0.9), length.quantile(0.95))
    # plt.hist(length)
    # plt.show()

    # 查看文本长度在 5% 分位点以下及 95% 分位点以上的 label 情况（长度较短的一端，新闻为真的比例更大，较长的一端，新闻为假的比例更大）
    print(data[data.text.apply(lambda x: len(x.decode('utf-8')) < length.quantile(0.05))].groupby(by='label').count())
    print(data[data.text.apply(lambda x: len(x.decode('utf-8')) < length.quantile(0.01))].groupby(by='label').count())
    print(data[data.text.apply(lambda x: len(x.decode('utf-8')) > length.quantile(0.95))].groupby(by='label').count())
    print(data[data.text.apply(lambda x: len(x.decode('utf-8')) > length.quantile(0.99))].groupby(by='label').count())


def get_char_seg(text):
    """
    对文本切分成单字，保留英文单词
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
    train_X = []
    test_X = []
    train_y = []
    test_y = []
    if (not train_size) and test_size:
        train_size = 1 - test_size
    for i in range(data.shape[0]):
        if i < train_size * 100:
            train_X.append(data.loc[i][X_name].decode('utf-8'))
            train_y.append(data.loc[i][y_name])
        else:
            test_X.append(data.loc[i][X_name].decode('utf-8'))
            test_y.append(data.loc[i][y_name])
    return train_X, test_X, train_y, test_y


def submit_data(model, seg_fun):
    input_path = os.path.join(ROOT_PATH, 'data/test_stage1.csv')
    out_path = os.path.join(ROOT_PATH, 'data/submit.csv')
    data = pd.read_csv(input_path)
    text_seg = [' '.join(item) for item in data.text.apply(lambda x: seg_fun(x))]
    labels = model.predict(text_seg)
    data_out = pd.DataFrame({'id': data.id, 'label': labels})
    data_out.to_csv(out_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    text = '你好呀 baby'
    print(get_char_seg(text))
