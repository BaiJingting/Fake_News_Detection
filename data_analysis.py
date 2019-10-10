# -*- coding: utf-8 -*-
import os
import json
import pandas as pd

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def analysis(data):
    """
    对数据做描述性统计分析，了解概况
    :return:
    """
    # # 了解样本整体情况
    # 是否有重复样本
    print(data.text.nunique(), data.shape[0])
    # # 查看重复样本文本情况
    # print(data.groupby(by='text').filter(lambda x: x.label.count() > 1).text)
    # # 是否存在文本相同、标注不同的情况
    # print(data.groupby(by=['text', 'label']).count().shape)
    # # 正负例是否均匀
    # print(data.groupby(by='label').count())
    # 缺失值处理（这里没有缺失值）
    print(data[data.isna().values])

    # # 了解文本情况
    length = data.text.apply(lambda x: len(x))
    print(length.describe())
    # 5%、10%、90%、95%、98%、99%分位数分别为：28、41、152、188、294、496
    print(length.quantile(0.05), length.quantile(0.1), length.quantile(0.9), length.quantile(0.95)
          , length.quantile(0.98), length.quantile(0.99))
    # plt.hist(length)
    # plt.show()

    # # 查看文本长度在 5% 分位点以下及 95% 分位点以上的 label 情况（长度较短的一端，新闻为真的比例更大，较长的一端，新闻为假的比例更大）
    # print(data[data.text.apply(lambda x: len(x.decode('utf-8')) < length.quantile(0.05))].groupby(by='label').count())
    # print(data[data.text.apply(lambda x: len(x.decode('utf-8')) < length.quantile(0.01))].groupby(by='label').count())
    # print(data[data.text.apply(lambda x: len(x.decode('utf-8')) > length.quantile(0.95))].groupby(by='label').count())
    # print(data[data.text.apply(lambda x: len(x.decode('utf-8')) > length.quantile(0.99))].groupby(by='label').count())


if __name__ == "__main__":
    # data = pd.read_csv(os.path.join(ROOT_PATH, 'data/train.csv'), encoding='utf-8')
    # test_data = pd.read_csv(os.path.join(ROOT_PATH, 'data/test_stage1.csv'), encoding='utf-8')
    # debunking_data = pd.read_csv(os.path.join(ROOT_PATH, 'data/debunking.csv'), encoding='utf-8')
    #
    # # # 对数据做描述性统计分析，了解概况
    # # analysis(data)
    # # analysis(test_data)
    # # analysis(debunking_data)
    #
    # print(data.text.nunique(), data.shape[0])
    # print(debunking_data.text.nunique(), debunking_data.shape[0])
    # text = pd.concat([data.text, debunking_data])
    # print(text.nunique(), text.shape)

    data = []
    with open(os.path.join(ROOT_PATH, 'data/rumors_v170613.json'), 'r') as f:
        for line in f.readlines():
            item = json.loads(line)['rumorText']
            data.append(item)
            print(item)
