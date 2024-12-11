#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_process.py
# @Author:
# @Date  : 2021/11/1 10:30
# @Desc  :
import json
import os
import random
import shutil

import numpy as np
from loguru import logger
import scipy.sparse as sp

def generate_all_interact():
    all_dict = {}
    files = ['click', 'cart', 'collect', 'buy']
    for file in files:
        with open(file+'_dict.txt') as r:
            data = json.load(r)
            for k, v in data.items():
                if all_dict.get(k, None) is None:
                    all_dict[k] = v
                else:
                    total = all_dict[k]
                    total.extend(v)
                    all_dict[k] = sorted(list(set(total)))
        with open('all_dict.txt', 'w') as w2:
            w2.write(json.dumps(all_dict))


def cold_start_sample():
    """
    1000 users cold start without delete the data under auxiliary behavior
    """
    with open('./origin/test_dict.txt', encoding='utf-8') as f, open('./origin/validation_dict.txt', encoding='utf-8') as val, open('./origin/buy_dict.txt', encoding='utf-8') as t:
        data = json.load(f)
        validate = json.load(val)
        buy_data = json.load(t)
        users = list(data.keys())
    test_users = np.random.choice(users, 1000, replace=False)
    test_dict = {k: data.pop(k) for k in test_users}
    [validate.pop(k, None) for k in test_users]
    [buy_data.pop(k, None) for k in test_users]

    with open('./origin/test_dict.txt', 'w') as w:
        w.write(json.dumps(test_dict))
    with open('./origin/buy_dict.txt', 'w') as w:
        w.write(json.dumps(buy_data))
    with open('./origin/validation_dict.txt', 'w') as w:
        w.write(json.dumps(validate))


def remove_test_data():
    """
    treat all test and validation date as cold start users, and remove these users' all interaction from buy behavior,
    meanwhile remove these user-item interactions from auxiliary behavior
    """
    origin = './origin'
    dest = '.'
    shutil.copyfile(os.path.join(origin, 'test_dict.txt'), os.path.join(dest, 'test_dict.txt'))
    shutil.copyfile(os.path.join(origin, 'validation_dict.txt'), os.path.join(dest, 'validation_dict.txt'))
    with open(os.path.join(origin, 'test_dict.txt'), encoding='utf-8') as t:
        test_data = json.load(t)
    with open(os.path.join(origin, 'validation_dict.txt'), encoding='utf-8') as t:
        validation_data = json.load(t)
    with open(os.path.join(origin, 'buy_dict.txt'), encoding='utf-8') as b:
        buy_data = json.load(b)
    for key in test_data.keys():
        buy_data.pop(key, None)
    # for key in validation_data.keys():
    #     buy_data.pop(key, None)
    with open(os.path.join(dest, 'buy_dict.txt'), 'w') as w1, open(os.path.join(dest, 'buy.txt'), 'w') as w2:
        w1.write(json.dumps(buy_data))
        for k, v in buy_data.items():
            for i in v:
                w2.write("{} {} \n".format(int(k), i))
    files = ['click', 'cart', 'collect']
    for file in files:
        with open(os.path.join(origin, file + '_dict.txt'), encoding='utf-8') as r:
            tmp_dict = json.load(r)
            for dic in [test_data, validation_data]:
                for key, val in dic.items():
                    tmp_val = tmp_dict.get(key, None)
                    if tmp_val is not None:
                        tmp_val = np.setdiff1d(tmp_val, val)
                        if len(tmp_val) < 1:
                            tmp_dict.pop(key)
                        else:
                            tmp_dict[key] = tmp_val.tolist()
            with open(os.path.join(dest, file + '_dict.txt'), 'w') as w1, open(os.path.join(dest, file + '.txt'), 'w') as w2:
                w1.write(json.dumps(tmp_dict))
                for k, v in tmp_dict.items():
                    for i in v:
                        w2.write("{} {} \n".format(int(k), i))


if __name__ == '__main__':
    path = '.'
    cold_start_sample()
    remove_test_data()
    generate_all_interact()
    # with open('buy_dict.txt') as f, open('buy.txt', 'w') as w:
    #     data = json.load(f)
    #     for k, v in data.items():
    #         for i in v:
    #             w.write("{} {} \n".format(int(k), i))

    with open('test_dict.txt', encoding='utf-8') as r1, open('buy_dict.txt') as r2:
        test = json.load(r1)
        train = json.load(r2)
        for k in test.keys():
            if train.get(k, None) is not None:
                print(k)
    #
    #
    files = ['click', 'cart', 'collect', 'buy']
    for file in files:
        with open('test_dict.txt', encoding='utf-8') as r1, open(file + '_dict.txt') as r2:
            test = json.load(r1)
            train = json.load(r2)
            for k, v in test.items():
                value = train.get(k, None)
                if value is not None:
                    length = len(list(set(v) & set(value)))
                    if length > 0:
                        print(k)