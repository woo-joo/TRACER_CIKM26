import torch
from torch import nn

from base_rec import Base_Recommender
from llm_enhancers.llm_enhancer import LLM_Enhancer



class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.m = args.m

        h = (args.d_llm + args.d // 2) // 2
        h = (h // args.m) * args.m

        self.w1 = nn.Linear(args.d_llm, h)
        self.activation = nn.LeakyReLU()
        self.w2 = nn.Conv1d(h,
                            args.m * args.d // 2,
                            kernel_size=1,
                            groups=args.m)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)


    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x[..., None])
        x = x.reshape(x.shape[0], self.m, -1)

        return x



class KAR(LLM_Enhancer):
    def __init__(self, args, base_rec: Base_Recommender):
        super().__init__(args, base_rec)

        self.batch_size = args.batch_size

        self.user_gate = nn.Linear(args.d_llm, 2 * args.m)
        self.item_gate = nn.Linear(args.d_llm, 2 * args.m)

        self.user_expert = MoE(args)
        self.item_expert = MoE(args)
        self.shared_expert = MoE(args)

        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.user_gate.weight)
        nn.init.xavier_uniform_(self.item_gate.weight)
        nn.init.zeros_(self.user_gate.bias)
        nn.init.zeros_(self.item_gate.bias)


    def __compute_moe__(self, x, entity):
        gate = getattr(self, f'{entity}_gate')
        expert = getattr(self, f'{entity}_expert')

        w = self.softmax(gate(x))[..., None]
        emb = torch.cat([expert(x), self.shared_expert(x)], dim=1)
        emb = torch.sum(w * emb, dim=1)

        return emb


    def update_H(self):
        self.base_rec.__propagate__()

        S_user = [self.__compute_moe__(self.X_user[i:i+self.batch_size], 'user')
                  for i in range(0, self.X_user.shape[0], self.batch_size)]
        S_user = torch.cat(S_user)

        S_item = [self.__compute_moe__(self.X_item[i:i+self.batch_size], 'item')
                  for i in range(0, self.X_item.shape[0], self.batch_size)]
        S_item = torch.cat(S_item)

        self.H_user = torch.cat([self.base_rec.H_user, S_user], dim=-1)
        self.H_item = torch.cat([self.base_rec.H_item, S_item], dim=-1)


    def forward(self, usr, pos, neg):
        out = self.base_rec(usr, pos, neg)

        S_usr = self.__compute_moe__(self.X_user[usr], 'user')
        S_pos = self.__compute_moe__(self.X_item[pos], 'item')
        S_neg = self.__compute_moe__(self.X_item[neg], 'item')
        
        out = {'H_usr': torch.cat([out['H_usr'], S_usr], dim=-1),
               'H_pos': torch.cat([out['H_pos'], S_pos], dim=-1),
               'H_neg': torch.cat([out['H_neg'], S_neg], dim=-1)}

        return out
