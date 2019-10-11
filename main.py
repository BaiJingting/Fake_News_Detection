# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

import process
# from models.fastText import FastTextClassify
from models.bert import BertClassify, DataGenerator, OurTokenizer


def fasttext():
    # # 预处理（分词 / 以字为单位）
    seg_fun = process.get_char_seg
    # seg_fun = preprocess.get_seg
    data['text_char_seg'] = data.text.apply(lambda x: seg_fun(x))
    train_data, test_data = process.split_train_test(data, 'text_char_seg', 'label', train_size=0.9)

    train_path = os.path.join(ROOT_PATH, 'data/fastText/train_file.txt')
    model_path = os.path.join(ROOT_PATH, 'model_files/fastText.model')

    fasttext_model = FastTextClassify(model_path)
    fasttext_model.save_train_file(train_data, train_path)
    fasttext_model.train(train_path)

    test_text = [' '.join(item[0]) for item in test_data]
    labels = fasttext_model.predict(test_text)

    # # 评估
    evaluate = classification_report(test_data[:, -1], labels)
    print(evaluate)

    process.submit_data(fasttext_model, seg_fun)


def bert(data, test_data):
    train_data, valid_data = process.split_train_test(data, 'text_handle', 'label', train_size=0.9)

    # # # bert
    # model_path = os.path.join(ROOT_PATH, 'model_files/bert/bert.h5')
    # model = BertClassify(initial_bert_model=False, model_path=model_path)
    model = BertClassify(initial_bert_model=True)
    model.train(train_data, valid_data)

    predict_results = model.predict(test_data.text_handle)
    with open(os.path.join(ROOT_PATH, 'data/bert/predict.txt'), 'w') as f:
        for i in range(test_data.shape[0]):
            label = 1 if predict_results[i][0] > 0.5 else 0
            f.write(test_data.id[i] + '\t' + test_data.text[i] + '\t' + str(predict_results[i][0])
                    + '\t' + str(label) + '\n')


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(ROOT_PATH, 'data/train.csv'), encoding='utf-8')
    test_data = pd.read_csv(os.path.join(ROOT_PATH, 'data/test_stage1.csv'), encoding='utf-8')

    data['text_handle'] = data.text.map(process.pretreatment)
    test_data['text_handle'] = test_data.text.map(process.pretreatment)

    bert(data, test_data)
