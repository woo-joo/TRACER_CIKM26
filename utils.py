import os
import random
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye, diags
import torch
from torch.utils.data import DataLoader

from sampler import TrainSampler, EvalSampler



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def read_data(args):
    data_df = pd.read_csv(f'datasets/{args.dataset[:-1]}.csv')
    data_df[['user', 'item']] = data_df[['user', 'item']] - 1
    data_df = data_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    for i in range(4):
        data = {}
        for user, items in data_df[data_df['block'] == i].groupby('user')['item']:
            items = items.to_list()
            if len(items) < 3:
                data[user] = [items, [], []]
                continue
            r = max(int(len(items) * 0.1), 1)
            data[user] = [items[:-2*r], items[-2*r:-r], items[-r:]]
        setattr(args, f'data{i}', data)

    idx = int(args.dataset[-1])
    data = getattr(args, f'data{idx}')
    if args.cl_frame == 'Full_Batch':
        for i in range(idx):
            for user, items in getattr(args, f'data{i}').items():
                if user in data:
                    data[user][0] += items[0]
                else:
                    data[user] = items


    pre = data_df[data_df['block'] == idx - 1]
    cur = data_df[data_df['block'] == idx]
    nxt = data_df[data_df['block'] == idx + 1]
    all = data_df


    args.data = data
    args.num_user_pre, args.num_item_pre = (pre['user'].max() + 1, pre['item'].max() + 1) if idx != 0 and args.cl_frame != 'Full_Batch' else (0, 0)
    args.num_user_cur, args.num_item_cur = (cur['user'].max() + 1, cur['item'].max() + 1)
    args.num_user_nxt, args.num_item_nxt = (nxt['user'].max() + 1, nxt['item'].max() + 1) if idx != 3 else \
                                           (cur['user'].max() + 1, cur['item'].max() + 1)
    args.num_user_all, args.num_item_all = (all['user'].max() + 1, all['item'].max() + 1)



def make_rating_matrix(args, split):
    row, col = [], []
    for user, items in args.data.items():
        if split == 'valid': items = items[0]
        if split == 'test': items = items[0] + items[1]
        row += [user] * len(items)
        col += items
    rating_matrix = csr_matrix(([1] * len(row), (row, col)),
                               (args.num_user_all, args.num_item_all))

    setattr(args, f'{split}_rating_matrix', rating_matrix)



def make_norm_adj_mat(args, i=False):
    row, col = [], []
    for user, (items, _, _) in args.data.items():
        row += [user] * len(items)
        col += [item + args.num_user_all for item in items]

    adj_row = row + col
    adj_col = col + row
    adj_val = [1] * (len(row) + len(col))
    size = (args.num_user_all + args.num_item_all,
            args.num_user_all + args.num_item_all)
    adj_mat = csr_matrix((adj_val, (adj_row, adj_col)), size)
    if i: adj_mat = adj_mat + eye(size[0], format='csr')

    degree = np.array(adj_mat.sum(axis=1)).flatten()
    degree_mat = diags(np.power(degree, -1.0 if i else -0.5, where=degree != 0))

    if i: norm_adj_mat = degree_mat @ adj_mat
    else: norm_adj_mat = degree_mat @ adj_mat @ degree_mat

    coo = norm_adj_mat.tocoo()
    row = torch.tensor(coo.row).long()
    col = torch.tensor(coo.col).long()
    index = torch.stack([row, col])
    value = torch.tensor(coo.data).float()

    norm_adj_mat = torch.sparse_coo_tensor(index, value, size)

    if i: args.norm_adj_mat_i = norm_adj_mat
    else: args.norm_adj_mat = norm_adj_mat



def get_dataloader(args):
    read_data(args)
    make_rating_matrix(args, 'valid')
    make_rating_matrix(args, 'test')
    make_norm_adj_mat(args)
    make_norm_adj_mat(args, i=True)

    train_set = TrainSampler(args.data, args.num_item_cur)
    valid_set = EvalSampler(args.data, 1)
    test_set = EvalSampler(args.data, 2)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size // 8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size // 8)
    
    return train_loader, valid_loader, test_loader



def get_model(args):
    from base_rec import Base_Recommender
    base_rec = Base_Recommender(args)

    if args.tracer:
        from tracer import TRACER
        cl_frame = TRACER(args, base_rec).to(args.device)

    else:
        llm_enhancer_module = import_module(f'llm_enhancers.{args.llm_enhancer.lower()}')
        llm_enhancer = getattr(llm_enhancer_module, args.llm_enhancer)(args, base_rec)

        cl_frame_name = args.cl_frame if args.cl_frame in ['ReLoop2', 'PISA'] else 'No'
        cl_frame_module = import_module(f'cl_frames.{cl_frame_name.lower()}')
        cl_frame = getattr(cl_frame_module, cl_frame_name)(args, llm_enhancer).to(args.device)

    return cl_frame
