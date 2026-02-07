import torch
from torch import nn



class Base_Recommender(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_user_all = args.num_user_all
        self.l = args.l if args.base_rec == 'LightGCN' else 0
        self.norm_adj_mat = args.norm_adj_mat.to(args.device)

        if args.llm_enhancer in ['KAR', 'LLM_ESR', 'TRACER']: d = args.d // 2
        else: d = args.d

        self.E_user = nn.Embedding(args.num_user_all, d)
        self.E_item = nn.Embedding(args.num_item_all, d)

        nn.init.xavier_uniform_(self.E_user.weight)
        nn.init.xavier_uniform_(self.E_item.weight)


    def __propagate__(self):
        E = torch.cat([self.E_user.weight, self.E_item.weight])
        Es = [E]
        for _ in range(self.l):
            E = self.norm_adj_mat @ E
            Es.append(E)
        Es = torch.stack(Es)
        E = torch.mean(Es, dim=0)

        self.H_user = E[:self.num_user_all]
        self.H_item = E[self.num_user_all:]


    def forward(self, usr, pos, neg):
        self.__propagate__()

        out = {'H_usr': self.H_user[usr],
               'H_pos': self.H_item[pos],
               'H_neg': self.H_item[neg]}

        return out


    def compute_loss(self, out):
        pos_score = torch.sum(out['H_usr'] * out['H_pos'], dim=-1)
        neg_score = torch.sum(out['H_usr'] * out['H_neg'], dim=-1)

        rec_loss = -torch.sum(torch.log(torch.sigmoid(pos_score - neg_score)))

        return rec_loss
