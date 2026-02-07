import os
import shutil
import argparse
from tqdm import tqdm
from copy import deepcopy

import torch
from torch import optim

from utils import set_seed, get_dataloader, get_model
from metrics import metrics, metric_funcs



def train(train_loader, model, optimizer, device):
    model.train()
    if hasattr(model, 'update_preference'): model.update_preference()
    train_loader.dataset.negative_sampling()

    for usr, pos, neg in train_loader:
        usr, pos, neg = usr.to(device), pos.to(device), neg.to(device)

        optimizer.zero_grad()

        out = model(usr, pos, neg)
        loss = model.compute_loss(out)

        loss.backward()
        optimizer.step()



@torch.no_grad()
def eval(eval_loader, model, rating_matrix, device):
    model.eval()
    model.llm_enhancer.update_H()

    ranks, users = [], []
    for usr, pos in eval_loader:
        usr, pos = usr.to(device), pos.to(device)

        logit = model.predict(usr)
        logit[rating_matrix[usr.cpu()-1].toarray() == 1] = 0

        rank = torch.argsort(torch.argsort(logit, descending=True))
        rank = rank[torch.arange(pos.shape[0]).to(device), pos]

        ranks.append(rank)
        users.append(usr)
    rank = torch.cat(ranks)
    user = torch.cat(users)
    results = [100 * result for metric_func in metric_funcs for result in metric_func(rank, user)]

    return results



def run_block(args):
    cl_frame = args.cl_frame
    if args.dataset[-1] == '0': args.cl_frame = 'Full_Batch'
    if args.tracer: args.cl_frame = args.llm_enhancer = 'TRACER'
    args.device = 'cuda'
    set_seed(args.seed)


    train_loader, valid_loader, test_loader = get_dataloader(args)
    model = get_model(args)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)


    if args.tracer:
        config = f'{args.base_rec}-TRACER         '
        if args.dataset[-1] == '0': config = f'{args.base_rec}-TRACER-Pretrain'
    else:
        cl_frame_len = max(len(cl_frame), len('Pretrain'))
        cl_frame = 'Pretrain' if args.dataset[-1] == '0' else cl_frame
        config = f'{args.base_rec}-{args.llm_enhancer}-{cl_frame:<{cl_frame_len}}'
    pbar = tqdm(range(1, args.num_epoch+1), desc=f'[ {config} ] Epoch   0')
    best = {'result': float('-inf'), 'epoch': -1, 'state_dict': None}
    for epoch in pbar:
        if epoch > best['epoch'] + args.patience:
            break

        train(train_loader, model, optimizer, args.device)
        result = eval(valid_loader, model, args.valid_rating_matrix, args.device)

        if best['result'] < result[-1]:
            best = {'result': result[-1], 'epoch': epoch, 'state_dict': deepcopy(model.state_dict())}
            if args.cl_frame == 'ReLoop2':
                error_memory = {'hasher': deepcopy(model.hasher.cpu()),
                                'x': deepcopy(model.error_memory_x.cpu()),
                                'y': deepcopy(model.error_memory_y.cpu())}
                best['error_memory'] = error_memory

        pbar.set_description(f'[ {config} ] Epoch {epoch:>{len(str(args.num_epoch))}}')
        pbar.set_postfix_str(f'Best= {best['result']:7.4f}, ' +
                             f'Current= {result[-1]:7.4f}, ' +
                             f'Patience= {(epoch - best['epoch']):>{len(str(args.patience))}}/{args.patience}')


    model.load_state_dict(best['state_dict'])
    if args.cl_frame == 'ReLoop2':
        model.hasher = best['error_memory']['hasher'].to(args.device)
        model.error_memory_x = best['error_memory']['x'].to(args.device)
        model.error_memory_y = best['error_memory']['y'].to(args.device)
    result = eval(test_loader, model, args.test_rating_matrix, args.device)


    if hasattr(model, 'get_merged_state_dict'):
        best['state_dict'] = model.get_merged_state_dict()
    if args.tracer: weight_path = f'weights/{args.base_rec}-TRACER/{args.dataset}.pt'
    else: weight_path = f'weights/{args.base_rec}-{args.llm_enhancer}/{args.cl_frame}/{args.dataset}.pt'
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in best['state_dict'].items()}, weight_path)
    if args.cl_frame == 'ReLoop2':
        error_path = weight_path.replace(args.dataset, f'{args.dataset}_error')
        torch.save(best['error_memory'], error_path)


    return result



def run(args):
    results = []
    block_args = deepcopy(args)
    for i in range(4):
        if i == 0 and args.cl_frame == 'Full_Batch' and not args.tracer: continue
        block_args.dataset = f'{args.dataset}{i}'
        if i > 0 and args.cl_frame == 'PISA':
            block_args.cl_frame = 'PISA_F'
            run_block(block_args)
        block_args.cl_frame = args.cl_frame
        result = run_block(block_args)
        results.append(result)

    results = torch.tensor(results[-3:])
    for i in range(1, 4):
        result_str = ', '.join([f'{metric}: {value:7.4f}' for metric, value in zip(metrics, results[i-1])])
        print(f'Block {i}: {result_str}')
    result = torch.mean(results, dim=0)
    result_str = ', '.join([f'{metric}: {value:7.4f}' for metric, value in zip(metrics, result)])
    print(f'Average: {result_str}')

    shutil.rmtree('weights')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_rec', type=str, default='LightGCN',
                        choices=['MF', 'LightGCN'])
    parser.add_argument('--llm_enhancer', type=str, default='No',
                        choices=['No', 'RLMRec', 'LLM2X', 'KAR', 'LLM_ESR'])
    parser.add_argument('--cl_frame', type=str, default='Full_Batch',
                        choices=['Full_Batch', 'Fine_Tune', 'ReLoop2', 'PISA'])
    parser.add_argument('--tracer', action='store_true')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['amazon-home', 'amazon-cds', 'amazon-movies', 'amazon-electronics', 'yelp'])
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--l', type=int, default=3)

    parser.add_argument('--w_sync', type=float, default=0.01, help='for RLMRec, LLM_ESR, TRACER')
    parser.add_argument('--m', type=int, default=4, help='for KAR')
    parser.add_argument('--n', type=int, default=4, help='for LLM_ESR')

    parser.add_argument('--k', type=int, default=60, help='for ReLoop2')
    parser.add_argument('--lambd', type=float, default=0.5, help='for ReLoop2')
    parser.add_argument('--r', type=float, default=1.0, help='for PISA')
    parser.add_argument('--w_cl', type=float, default=1.0, help='for PISA')

    parser.add_argument('--lora_r', type=int, default=8, help='for TRACER')
    parser.add_argument('--lora_alpha', type=int, default=8, help='for TRACER')

    args = parser.parse_args()

    run(args)
