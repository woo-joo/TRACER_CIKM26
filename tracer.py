from copy import deepcopy

import torch
from torch import nn

from base_rec import Base_Recommender
from llm_enhancers.llm_enhancer import LLM_Enhancer
from cl_frames.cl_frame import CL_Frame



class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, pretrain=False):
        super().__init__()

        self.scale = alpha / r

        self.linear = nn.Linear(in_features, out_features)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        if pretrain:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            nn.init.zeros_(self.A.weight)
            nn.init.zeros_(self.B.weight)

            self.A.weight.requires_grad = False
            self.B.weight.requires_grad = False
        else:
            nn.init.xavier_uniform_(self.A.weight)
            nn.init.zeros_(self.B.weight)

            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False


    @torch.no_grad()
    def merge(self):
        self.linear.weight.data += (self.B.weight @ self.A.weight) * self.scale


    def forward(self, x):
        return self.linear(x) + self.B(self.A(x)) * self.scale



class TRACER_LLM_Enhacer(LLM_Enhancer):
    def __init__(self, args, base_rec: Base_Recommender):
        super().__init__(args, base_rec)

        self.batch_size = args.batch_size

        self.gate = nn.Sequential(nn.Linear(args.d_llm, 1),
                                  nn.Sigmoid())

        h = (args.d_llm + args.d // 2) // 2
        kwargs = {'r'       : args.lora_r,
                  'alpha'   : args.lora_alpha,
                  'pretrain': args.dataset[-1] == '0'}
        self.adapter = nn.Sequential(LoRALinear(args.d_llm, h, **kwargs),
                                     nn.LeakyReLU(),
                                     LoRALinear(h, args.d // 2, **kwargs))

        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)


    @torch.no_grad()
    def __init_llm_emb__(self, args):
        new_user = slice(args.num_user_pre, args.num_user_cur)
        new_item = slice(args.num_item_pre, args.num_item_cur)

        if args.dataset[-1] == '0':
            X = torch.cat([self.X_user[new_user], self.X_item[new_item]])
            X = X - torch.mean(X, dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(X, full_matrices=False)

            V = Vh.T[:, :args.d//2]
            S = X @ V

        else:
            E_user = self.base_rec.E_user.weight.data.detach().clone().to(args.device)
            E_item = self.base_rec.E_item.weight.data.detach().clone().to(args.device)

            E_user[args.num_user_pre:, :] = 0.0
            E_item[args.num_item_pre:, :] = 0.0

            E_pre = torch.cat([E_user[:args.num_user_pre], E_item[:args.num_item_pre]])
            X_pre = torch.cat([self.X_user[:args.num_user_pre], self.X_item[:args.num_item_pre]])
            M = E_pre.T @ X_pre
            U, _, Vh = torch.linalg.svd(M, full_matrices=False)
            P = Vh.T @ U.T

            S_user_se = self.X_user @ P
            S_item_se = self.X_item @ P

            norm_adj_mat = args.norm_adj_mat_i.to(args.device)
            S_user_co = (norm_adj_mat @ torch.cat([S_user_se, E_item]))[:args.num_user_all]
            S_item_co = (norm_adj_mat @ torch.cat([E_user, S_item_se]))[args.num_user_all:]

            S_se = torch.cat([S_user_se[new_user], S_item_se[new_item]])
            S_co = torch.cat([S_user_co[new_user], S_item_co[new_item]])

            beta = (nn.functional.cosine_similarity(S_se, S_co, dim=-1) + 1) / 2
            S = beta[:, None] * S_se + (1 - beta)[:, None] * S_co

        S_user = S[:(args.num_user_cur - args.num_user_pre)]
        S_item = S[(args.num_user_cur - args.num_user_pre):]

        self.base_rec.E_user.weight.data[new_user] = S_user
        self.base_rec.E_item.weight.data[new_item] = S_item


    def update_H(self):
        self.base_rec.__propagate__()

        S_user = []
        for i in range(0, self.X_user.shape[0], self.batch_size):
            X_user = self.X_user[i:i+self.batch_size]
            S_user.append(self.gate(X_user) * self.adapter(X_user))
        S_user = torch.cat(S_user)

        S_item = []
        for i in range(0, self.X_item.shape[0], self.batch_size):
            X_item = self.X_item[i:i+self.batch_size]
            S_item.append(self.gate(X_item) * self.adapter(X_item))
        S_item = torch.cat(S_item)

        self.H_user = torch.cat([self.base_rec.H_user, S_user], dim=-1)
        self.H_item = torch.cat([self.base_rec.H_item, S_item], dim=-1)


    def forward(self, usr, pos, neg):
        out = self.base_rec(usr, pos, neg)

        E_usr = out['H_usr']
        E_pos = out['H_pos']
        E_neg = out['H_neg']

        c_usr = self.gate(self.X_user[usr])
        c_pos = self.gate(self.X_item[pos])
        c_neg = self.gate(self.X_item[neg])

        S_usr = self.adapter(self.X_user[usr])
        S_pos = self.adapter(self.X_item[pos])
        S_neg = self.adapter(self.X_item[neg])
        
        H_usr = torch.cat([E_usr, c_usr * S_usr], dim=-1)
        H_pos = torch.cat([E_pos, c_pos * S_pos], dim=-1)
        H_neg = torch.cat([E_neg, c_neg * S_neg], dim=-1)

        out = {'usr': usr, 'pos': pos, 'neg': neg,
               'E_usr': E_usr, 'E_pos': E_pos, 'E_neg': E_neg,
               'S_usr': S_usr, 'S_pos': S_pos, 'S_neg': S_neg,
               'H_usr': H_usr, 'H_pos': H_pos, 'H_neg': H_neg}

        return out



class TRACER(CL_Frame):
    def __init__(self, args, base_rec: Base_Recommender):
        llm_enhancer = TRACER_LLM_Enhacer(args, base_rec)

        super().__init__(args, llm_enhancer)

        self.w_sync = args.w_sync
        self.num_user_pre = args.num_user_pre
        self.num_item_pre = args.num_item_pre

        self.ce_loss = nn.CrossEntropyLoss()
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        self.__load_prev__(args)
        self.llm_enhancer.__init_with_X__(args)


    def __load_prev__(self, args):
        idx = int(args.dataset[-1])
        if idx == 0:
            self.E_user_pre = torch.zeros(args.num_user_all, args.d // 2).to(args.device)
            self.E_item_pre = torch.zeros(args.num_item_all, args.d // 2).to(args.device)
            return

        self.to(args.device)

        weight_path = f'weights/{args.base_rec}-TRACER/{args.dataset[:-1]}{idx-1}.pt'
        state_dict = torch.load(weight_path)
        self.load_state_dict(state_dict, strict=False)
        self.llm_enhancer.update_H()
        self.E_user_pre = self.llm_enhancer.base_rec.H_user.detach().clone()
        self.E_item_pre = self.llm_enhancer.base_rec.H_item.detach().clone()

        self.to('cpu')


    @torch.no_grad()
    def get_merged_state_dict(self):
        state_dict = deepcopy(self.state_dict())

        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                linear_key = f'{name}.linear.weight'
                A_key = f'{name}.A.weight'
                B_key = f'{name}.B.weight'

                delta = (state_dict[B_key] @ state_dict[A_key]) * module.scale
                state_dict[linear_key] = state_dict[linear_key] + delta

                del state_dict[A_key], state_dict[B_key]

        return state_dict


    def __compute_info_nce_loss__(self, idx, z_i, z_j):
        mask = (idx[:, None] == idx[None, :])
        mask.fill_diagonal_(0)

        z_i = nn.functional.normalize(z_i)
        z_j = nn.functional.normalize(z_j)

        logit = z_i @ z_j.T
        logit[mask] = -10000.0
        label = torch.arange(logit.shape[0]).to(logit.device)

        info_nce_loss = self.ce_loss(logit, label)

        return info_nce_loss


    def compute_loss(self, out):
        rec_loss = self.llm_enhancer.base_rec.compute_loss(out)

        guide_loss = self.__compute_info_nce_loss__(out['usr'], out['E_usr'], out['S_usr']) + \
                     self.__compute_info_nce_loss__(out['pos'], out['E_pos'], out['S_pos']) + \
                     self.__compute_info_nce_loss__(out['neg'], out['E_neg'], out['S_neg'])

        sync_loss = 0.0
        iterator = zip(
            [out['usr'], out['pos'], out['neg']],
            [out['E_usr'], out['E_pos'], out['E_neg']],
            [out['S_usr'], out['S_pos'], out['S_neg']],
            [self.E_user_pre, self.E_item_pre, self.E_item_pre],
            [self.num_user_pre, self.num_item_pre, self.num_item_pre]
        )
        for id, E, S, E_pre, num_pre in iterator:
            g_rec = torch.autograd.grad(rec_loss, E, retain_graph=True)[0]
            g_guide = torch.autograd.grad(guide_loss, E, retain_graph=True)[0]
            cos_sim = self.cos_sim(g_rec, g_guide)

            mask_guide = (cos_sim > 0).float()[:, None]
            mask_reg = ((cos_sim <= 0) & (id < num_pre)).float()[:, None]

            sync_guide_loss = self.__compute_info_nce_loss__(id, E * mask_guide, S * mask_guide)
            sync_reg_loss = self.__compute_info_nce_loss__(id, E * mask_reg, E_pre[id] * mask_reg)
            sync_loss += sync_guide_loss + sync_reg_loss

        return rec_loss + self.w_sync * sync_loss
