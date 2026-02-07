import torch
from torch import nn

from base_rec import Base_Recommender



class LLM_Enhancer(nn.Module):
    def __init__(self, args, base_rec: Base_Recommender):
        super().__init__()

        self.base_rec = base_rec

        self.__load_X__(args)


    def __load_X__(self, args):
        emb = torch.load(f'sem_reprs/{args.dataset[:-1]}.pt')[args.dataset[-1]]
        args.d_llm = emb['user'].shape[1]

        self.X_user = torch.zeros(args.num_user_all, args.d_llm).to(args.device)
        self.X_item = torch.zeros(args.num_item_all, args.d_llm).to(args.device)

        self.X_user[:emb['user'].shape[0]] = emb['user'].to(args.device)
        self.X_item[:emb['item'].shape[0]] = emb['item'].to(args.device)


    @torch.no_grad()
    def __init_with_X__(self, args):
        if args.llm_enhancer not in ['LLM2X', 'LLM_ESR']: return

        new_user = slice(args.num_user_pre, args.num_user_cur)
        new_item = slice(args.num_item_pre, args.num_item_cur)

        X = torch.cat([self.X_user[new_user], self.X_item[new_item]])
        X = X - torch.mean(X, dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)

        d = args.d if args.llm_enhancer == 'LLM2X' else args.d // 2
        V = Vh.T[:, :d]
        S = X @ V

        S_user = S[:(args.num_user_cur - args.num_user_pre)]
        S_item = S[(args.num_user_cur - args.num_user_pre):]

        self.base_rec.E_user.weight.data[new_user] = S_user
        self.base_rec.E_item.weight.data[new_item] = S_item


    def update_H(self):
        self.base_rec.__propagate__()

        self.H_user = self.base_rec.H_user
        self.H_item = self.base_rec.H_item


    def forward(self, usr, pos, neg): return self.base_rec(usr, pos, neg)


    def predict(self, usr):
        H_user = self.H_user[usr]
        H_item = self.H_item
        logit = H_user @ H_item.T

        return logit


    def compute_loss(self, out): return self.base_rec.compute_loss(out)
