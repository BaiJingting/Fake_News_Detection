# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

import process
from models.fastText import FastTextClassify
from models.bert import BertClassify


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(ROOT_PATH, 'data/train.csv'), encoding='utf-8')
    test_data = pd.read_csv(os.path.join(ROOT_PATH, 'data/test_stage1.csv'), encoding='utf-8')

    # # 对数据做描述性统计分析，了解概况
    # process.analysis(data)

    train_X, valid_X, train_y, valid_y = process.split_train_test(data, 'text', 'label', train_size=0.85)

    # # # fasttext

    # # # 预处理（分词 / 以字为单位）
    # seg_fun = process.get_char_seg
    # # seg_fun = preprocess.get_seg
    # data['text_char_seg'] = data.text.apply(lambda x: seg_fun(x))
    # train_X, test_X, train_y, test_y = process.split_train_test(data, 'text_char_seg', 'label', train_size=0.85)
    #
    # train_path = os.path.join(ROOT_PATH, 'data/fastText/train_file.txt')
    # test_path = os.path.join(ROOT_PATH, 'data/fastText/test_file.txt')
    # model_path = os.path.join(ROOT_PATH, 'model_files/fastText.model')
    #
    # fasttext_model = FastTextClassify(model_path)
    # fasttext_model.save_train_file(train_X, train_y, train_path)
    # fasttext_model.train(train_path)
    #
    # test_text = [' '.join(item) for item in test_X]
    # labels = fasttext_model.predict(test_text)

    # # # bert
    in_model_path = os.path.join(ROOT_PATH, 'model_files/bert/chinese_L-12_H-768_A-12')
    out_model_path = os.path.join(ROOT_PATH, 'model_files/bert')
    model = BertClassify(train=True)
    model.train(train_X, valid_X)

    predict_results, labels = model.predict(test_data)
    with open(os.path.join(ROOT_PATH, 'data/bert/predict.txt'), 'w') as f:
        for i in range(test_data.shape[0]):
            f.write(test_data.id[0].encode('utf-8') + '\t' + str(labels[i]) + '\n')

    # # # 评估
    # evaluate = classification_report(test_y, labels)
    # print(evaluate)

    # process.submit_data(fasttext_model, seg_fun)
