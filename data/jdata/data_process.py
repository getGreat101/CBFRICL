#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_process.py
# @Author: yanms
# @Date  : 2021/11/23 20:04
# @Desc  :
import json
import os
import random
import shutil

import pandas as pd
from collections import Counter
import numpy as np

from loguru import logger
import scipy.sparse as sp


def _data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     划分比例
    :param shuffle:   是否打乱
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return full_list, []
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def get_statistics_file():
    # 1=浏览；2=下单；3=关注；4=评论；5=加购
    df = pd.read_csv('jdata_action.csv', usecols=[0, 1, 4])
    df = df.drop_duplicates()
    df = df.drop(df[df['type'] == 4].index)

    # 过滤掉购买次数少于1次的数据--用户
    buy_view = df[df.iloc[:, 2] == 2]
    buy_ids = buy_view.groupby('user_id').filter(lambda x: (len(x) >= 3))['user_id'].to_numpy()
    buy_ids = np.unique(buy_ids)
    print("过滤购买后，{}".format(len(buy_ids)))
    df = df[df['user_id'].isin(buy_ids)]

    # 过滤掉购买次数少于1次的数据-- 物品
    item_view = df[df.iloc[:, 2] == 2]
    item_ids = item_view.groupby('sku_id').filter(lambda x: (len(x) >= 3))['sku_id'].to_numpy()
    item_ids = np.unique(item_ids)
    print("过滤购买后，{}".format(len(item_ids)))

    data = df[df['sku_id'].isin(item_ids)]


    # cart_view = df[df.iloc[:, 2] == 5]
    # cart_ids = cart_view.groupby('user_id').filter(lambda x: (len(x) > 0))['user_id'].to_numpy()
    # cart_ids = np.unique(cart_ids)
    # print("过滤加购物车后，{}".format(len(cart_ids)))

    # collect_view = df[df.iloc[:, 2] == 3]
    # collect_ids = collect_view.groupby('user_id').filter(lambda x: (len(x) > 0))['user_id'].to_numpy()
    # collect_ids = np.unique(collect_ids)
    # print("过滤收藏后，{}".format(len(collect_ids)))

    # view_view = df[df.iloc[:, 2] == 1]
    # view_ids = view_view.groupby('user_id').filter(lambda x: (len(x) > 0 & len(x) < 800))['user_id'].to_numpy()
    # view_ids = np.unique(view_ids)
    # print("过滤浏览后，{}".format(len(view_ids)))

    # ids = np.union1d(cart_ids, collect_ids)
    # ids = np.union1d(ids, view_ids)
    # ids = np.unique(ids)
    # ids = np.intersect1d(ids, buy_ids)
    # ids = np.unique(ids)
    # df = df[df['user_id'].isin(ids)]

    # view_view = df[df.iloc[:, 2] == 1]
    # view_item = view_view.groupby('sku_id').filter(lambda x: (len(x) < 20))['sku_id'].to_numpy()
    # view_item = np.unique(view_item)
    # print("过滤浏览后item，{}".format(len(view_item)))

    # data = df[~df['sku_id'].isin(view_item)]


    users = sorted(list(set(data.iloc[:, 0].tolist())))
    items = sorted(list(set(data.iloc[:, 1].tolist())))

    print("最终用户数，{}".format(len(users)))
    print("最终商品数，{}".format(len(items)))

    with open('count.txt', 'w') as c:
        dic = {'user': len(users), 'item': len(items)}
        c.write(json.dumps(dic))
    user_dic = dict(zip(users, [x for x in range(len(users))]))
    item_dic = dict(zip(items, [x for x in range(len(items))]))
    data.iloc[:, 0] = data.iloc[:, 0].map(user_dic)
    data.iloc[:, 1] = data.iloc[:, 1].map(item_dic)
    data = data.sort_values(by='user_id', ascending=True)
    data_list = data.to_numpy().tolist()
    view_dict = {}
    all_buy_dict = {}
    collect_dict = {}
    cart_dict = {}
    for user, item, behavior in data_list:
        user, item = user + 1, item + 1
        if behavior == 1:
            if view_dict.get(user, None) is None:
                view_dict[user] = [item]
            if item not in view_dict[user]:
                view_dict[user].append(item)
        elif behavior == 2:
            if all_buy_dict.get(user, None) is None:
                all_buy_dict[user] = [item]
            if item not in all_buy_dict[user]:
                all_buy_dict[user].append(item)
        elif behavior == 3:
            if collect_dict.get(user, None) is None:
                collect_dict[user] = [item]
            if item not in collect_dict[user]:
                collect_dict[user].append(item)
        elif behavior == 5:
            if cart_dict.get(user, None) is None:
                cart_dict[user] = [item]
            if item not in cart_dict[user]:
                cart_dict[user].append(item)
    train_dict = {}
    test_dict = {}
    validation_dict = {}
    for k, v in all_buy_dict.items():
        tmp1, tmp2 = _data_split(v, 0.8)
        train_dict[k] = tmp1
        if len(tmp2) > 0:
            test_l, valid_l = _data_split(tmp2, 0.5)
            validation_dict[k] = test_l
            if len(valid_l) > 0:
                test_dict[k] = valid_l

    # for k, v in validation_dict.items():
    #     view_val = view_dict.get(k, None)
    #     if view_val is not None:
    #         view_val = np.setdiff1d(np.array(view_val), np.array(v)).tolist()
    #         if len(view_val) == 0:
    #             view_dict.pop(k)
    #         else:
    #             view_dict[k] = view_val
    #     cart_val = cart_dict.get(k, None)
    #     if cart_val is not None:
    #         cart_val = np.setdiff1d(np.array(cart_val), np.array(v)).tolist()
    #         if len(cart_val) == 0:
    #             cart_dict.pop(k)
    #         else:
    #             cart_dict[k] = cart_val
    #     col_val = collect_dict.get(k, None)
    #     if col_val is not None:
    #         col_val = np.setdiff1d(np.array(col_val), np.array(v)).tolist()
    #         if len(col_val) == 0:
    #             collect_dict.pop(k)
    #         else:
    #             collect_dict[k] = col_val
    
    # for k, v in test_dict.items():
    #     view_val = view_dict.get(k, None)
    #     if view_val is not None:
    #         view_val = np.setdiff1d(np.array(view_val), np.array(v)).tolist()
    #         if len(view_val) == 0:
    #             view_dict.pop(k)
    #         else:
    #             view_dict[k] = view_val
    #     cart_val = cart_dict.get(k, None)
    #     if cart_val is not None:
    #         cart_val = np.setdiff1d(np.array(cart_val), np.array(v)).tolist()
    #         if len(cart_val) == 0:
    #             cart_dict.pop(k)
    #         else:
    #             cart_dict[k] = cart_val
    #     col_val = collect_dict.get(k, None)
    #     if col_val is not None:
    #         col_val = np.setdiff1d(np.array(col_val), np.array(v)).tolist()
    #         if len(col_val) == 0:
    #             collect_dict.pop(k)
    #         else:
    #             collect_dict[k] = col_val

    with open('view_dict.txt', 'w') as v, open('cart_dict.txt', 'w') as c, open('collect_dict.txt', 'w') as cc, open('all_buy_dict.txt', 'w') as b:
        v.write(json.dumps(view_dict))
        b.write(json.dumps(cart_dict))
        cc.write(json.dumps(collect_dict))
        c.write(json.dumps(all_buy_dict))

    with open('train_dict.txt', 'w') as v, open('test_dict.txt', 'w') as c, open('validation_dict.txt', 'w') as b:
        v.write(json.dumps(train_dict))
        b.write(json.dumps(validation_dict))
        c.write(json.dumps(test_dict))

    with open('view.txt', 'w') as f1, \
            open('cart.txt', 'w') as f2, \
            open('collect.txt', 'w') as f6, \
            open('train.txt', 'w') as f3, \
            open('test.txt', 'w') as f4, \
            open('validation.txt', 'w') as f5:
        for k, v in view_dict.items():
            for i in v:
                f1.write('{} {}\n'.format(k, i))
        for k, v in cart_dict.items():
            for i in v:
                f2.write('{} {}\n'.format(k, i))
        for k, v in collect_dict.items():
            for i in v:
                f6.write('{} {}\n'.format(k, i))
        for k, v in train_dict.items():
            for i in v:
                f3.write('{} {}\n'.format(k, i))
        for k, v in test_dict.items():
            for i in v:
                f4.write('{} {}\n'.format(k, i))
        for k, v in validation_dict.items():
            for i in v:
                f5.write('{} {}\n'.format(k, i))

    all_interact = {}
    for dic in [train_dict, view_dict, cart_dict, collect_dict]:
        for k, v in dic.items():
            if all_interact.get(k, None) is None:
                all_interact[k] = v
            else:
                all_interact[k].extend(v)
    for k, v in all_interact.items():
        all_interact[k] = sorted(list(set(v)))
    with open('all_dict.txt', 'w') as f:
        f.write(json.dumps(all_interact))

    shutil.copyfile('train.txt', 'buy.txt')
    shutil.copyfile('train_dict.txt', 'buy_dict.txt')


def train_sample():
    samples = []
    with open('count.txt') as r:
        count_info = json.load(r)
    item_count = count_info['item']
    item_list = np.array([x + 1 for x in range(item_count)])
    with open('buy_dict.txt') as r, open('train_samples.txt', 'w') as w:
        data = json.load(r)
        for k, v in data.items():
            k = int(k)
            value_list = np.array(v)
            negatives = np.random.choice(np.setdiff1d(item_list, value_list), len(v) * 5).tolist()
            for value in v:
                samples.append([k, value, 1.0])
            for neg in negatives:
                samples.append([k, neg, 0.0])
        w.write(json.dumps(samples))

def item_inter():
    with open('count.txt') as r:
        data = json.load(r)
        item_num = data['item']
    for behavior in ['view', 'cart', 'collect', 'buy']:
        all_inter = set()
        with open(behavior + '_dict.txt') as f:
            data = json.load(f)
            for v in data.values():
                i = len(v)
                m = 0
                while m < i:
                    n = 0
                    while n < i:
                        all_inter.add((v[m], v[n]))
                        n += 1
                    m += 1
        row = []
        col = []
        for item in all_inter:
            row.append(item[0])
            col.append(item[1])
        indict = len(row)
        item_graph = sp.coo_matrix((np.ones(indict), (row, col)), shape=[item_num, item_num])
        item_graph_degree = item_graph.toarray().sum(axis=0).reshape(-1, 1)
        info = {'row': row, 'col': col, 'degree': item_graph_degree.tolist()}
        with open(behavior+'_item_graph.txt', 'w', encoding='utf-8') as f:
            f.write(json.dumps(info))

def generate_dict(path, file):
    user_interaction = {}
    with open(os.path.join(path, file)) as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            user, item = int(user), int(item)

            if user not in user_interaction:
                user_interaction[user] = [item]
            elif item not in user_interaction[user]:
                user_interaction[user].append(item)
    return user_interaction

@logger.catch()
def generate_interact(path):
    buy_dict = generate_dict(path, 'buy.txt')
    with open(os.path.join(path, 'buy_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(buy_dict))

    cart_dict = generate_dict(path, 'cart.txt')
    with open(os.path.join(path, 'cart_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(cart_dict))

    view_dict = generate_dict(path, 'view.txt')
    with open(os.path.join(path, 'view_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(view_dict))

    collect_dict = generate_dict(path, 'collect.txt')
    with open(os.path.join(path, 'collect_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(collect_dict))

    for dic in [buy_dict, cart_dict, collect_dict]:
        for k, v in dic.items():
            if k not in view_dict:
                view_dict[k] = v
            item = view_dict[k]
            item.extend(v)
    for k, v in view_dict.items():
        item = view_dict[k]
        item = list(set(item))
        view_dict[k] = sorted(item)

    with open(os.path.join(path, 'all_train_interact_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(view_dict))

    shutil.copyfile('buy_dict.txt', 'train_dict.txt')

    validation_dict = generate_dict(path, 'validation.txt')
    with open(os.path.join(path, 'validation_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(validation_dict))

    test_dict = generate_dict(path, 'test.txt')
    with open(os.path.join(path, 'test_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_dict))


def generate_all_interact():
    all_dict = {}
    files = ['view', 'cart', 'collect', 'buy']
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
        with open('all.txt', 'w') as w1, open('all_dict.txt', 'w') as w2:
            for k, v in all_dict.items():
                for i in v:
                    w1.write('{} {}\n'.format(int(k), i))
            w2.write(json.dumps(all_dict))

if __name__ == '__main__':
    print('start process...')

    # get_statistics_file()
    # train_sample()
    generate_interact("./")
    generate_all_interact()
    # item_inter()
    # file_name = ['view', 'cart', 'buy', 'test', 'validation', 'train']
    # for file in file_name:
    #     with open(file + '.txt') as r, open(file + '_.txt', 'w') as w:
    #         line = r.readlines()
    #         for l in line:
    #             l = l.strip('\n').strip().split(' ')
    #             w.write('{} {}\n'.format(int(l[0]) + 1, int(l[1]) + 1))
    # for file in file_name:
    #     os.rename(file+'_.txt', file+'.txt')
    # generate_interact('.')

    # with open('buy_dict.txt') as b, open('cart_dict.txt') as c, open('view_dict.txt') as v:
    #     buy = json.load(b)
    #     cart = json.load(c)
    #     view = json.load(v)
    #     arr = []
    #     for k, value in buy.items():
    #         bv = view.get(k, None)
    #         if bv is not None:
    #             for i in value:
    #                 if i not in bv:
    #                     arr.append(i)
    #         else:
    #             arr.extend(value)
    #     print(len(arr))
    # with open('collect_dict.txt') as b, open('cart_dict.txt') as c, open('view_dict.txt') as v:
    #     collect = json.load(b)
    #     cart = json.load(c)
    #     view = json.load(v)

    #     for i in list(collect):
    #         if len(collect[i]) == 0:
    #             collect.pop(i)
    #     for i in list(cart):
    #         if len(cart[i]) == 0:
    #             cart.pop(i)
    #     for i in list(view):
    #         if len(view[i]) == 0:
    #             view.pop(i)

    # with open('collect_dict.txt', 'w') as b, open('cart_dict.txt', 'w') as c, open('view_dict.txt', 'w') as v:
    #     b.write(json.dumps(collect))
    #     c.write(json.dumps(cart))
    #     v.write(json.dumps(view))



