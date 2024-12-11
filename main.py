import argparse
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from data_set import DataSet

from model_cascade import CBFRICL


from trainer import Trainer

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--excel_path', type=str, default='', help='')
    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=0.001, help='')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--node_dropout', type=float, default=0.75)
    parser.add_argument('--message_dropout', type=float, default=0.25)

    parser.add_argument('--data_name', type=str, default='tmall_cold', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')#控制是否加载模型参数

    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=3072, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=200, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')#保存模型地址
    parser.add_argument('--check_point', type=str, default='', help='')#加载模型地址
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cuda', help='')

    args = parser.parse_args()
    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'Tmall'
    elif args.data_name == 'tmall_cold':
        args.data_path = 'data/Tmall_cold_all'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'Tmall_cold_all'
    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        args.behaviors = ['view', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'beibei'
    elif args.data_name == 'beibei_cold':
        args.data_path = './data/beibei_cold_all'
        args.behaviors = ['view', 'cart', 'buy']
        args.layers = [1, 1, 1]
        args.model_name = 'beibei'
    elif args.data_name == 'jdata':
        args.data_path = './data/jdata'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
        args.layers = [1, 1, 1, 1]
        args.model_name = 'jdata'
    else:
        raise Exception('data_name cannot be None')

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME
    logfile = '{}_enb_{}_{}'.format(args.model_name, args.embedding_size, TIME)
    args.train_writer = SummaryWriter('../tf-logs/train/' + logfile)
    args.test_writer = SummaryWriter('../tf-logs/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model = CBFRICL(args, dataset)
    
#     model._load_model()
    
    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    logger.info('train end total cost time: {}'.format(time.time() - start))



