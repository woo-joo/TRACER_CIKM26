import torch
from torch import nn

from base_rec import Base_Recommender
from llm_enhancers.llm_enhancer import LLM_Enhancer



class LLM_ESR(LLM_Enhancer):
    def __init__(self, args, base_rec: Base_Recommender):
        super().__init__(args, base_rec)

        self.n = args.n
        self.w_guide = args.w_sync
        self.batch_size = args.batch_size

        self.adapter = nn.Sequential(nn.Linear(args.d_llm, (args.d_llm + args.d // 2) // 2),
                                     nn.LeakyReLU(),
                                     nn.Linear((args.d_llm + args.d // 2) // 2, args.d // 2))

        self.mse_loss = nn.MSELoss()

        nn.init.xavier_uniform_(self.adapter[0].weight)
        nn.init.xavier_uniform_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].bias)

        self.__retrieve_sim__()


    @torch.no_grad()
    def __retrieve_sim__(self):
        X_user = nn.functional.normalize(self.X_user)
        X_item = nn.functional.normalize(self.X_item)

        user_sim = []
        for i in range(0, X_user.shape[0], self.batch_size):
            user_sim_mat = X_user[i:i+self.batch_size] @ X_user.T
            user_sim_mat.fill_diagonal_(-10000.0)
            user_sim.append(torch.topk(user_sim_mat, self.n, dim=-1).indices)
        self.user_sim = torch.cat(user_sim)

        item_sim = []
        for i in range(0, X_item.shape[0], self.batch_size):
            item_sim_mat = X_item[i:i+self.batch_size] @ X_item.T
            item_sim_mat.fill_diagonal_(-10000.0)
            item_sim.append(torch.topk(item_sim_mat, self.n, dim=-1).indices)
        self.item_sim = torch.cat(item_sim)


    def update_H(self):
        self.base_rec.__propagate__()

        S_user = [self.adapter(self.X_user[i:i+self.batch_size])
                  for i in range(0, self.X_user.shape[0], self.batch_size)]
        S_user = torch.cat(S_user)

        S_item = [self.adapter(self.X_item[i:i+self.batch_size])
                  for i in range(0, self.X_item.shape[0], self.batch_size)]
        S_item = torch.cat(S_item)

        self.H_user = torch.cat([self.base_rec.H_user, S_user], dim=-1)
        self.H_item = torch.cat([self.base_rec.H_item, S_item], dim=-1)


    def forward(self, usr, pos, neg):
        usr = torch.cat([usr[:, None], self.user_sim[usr]], dim=1)
        pos = torch.cat([pos[:, None], self.item_sim[pos]], dim=1)
        neg = torch.cat([neg[:, None], self.item_sim[neg]], dim=1)

        out = self.base_rec(usr, pos, neg)

        S_usr = self.adapter(self.X_user[usr])
        S_pos = self.adapter(self.X_item[pos])
        S_neg = self.adapter(self.X_item[neg])
        
        H_usr = torch.cat([out['H_usr'], S_usr], dim=-1)
        H_pos = torch.cat([out['H_pos'], S_pos], dim=-1)
        H_neg = torch.cat([out['H_neg'], S_neg], dim=-1)

        out = {'H_usr': H_usr[:, 0],
               'H_pos': H_pos[:, 0],
               'H_neg': H_neg[:, 0],
               'H_usr_sim': torch.mean(H_usr[:, 1:], dim=1),
               'H_pos_sim': torch.mean(H_pos[:, 1:], dim=1),
               'H_neg_sim': torch.mean(H_neg[:, 1:], dim=1)}

        return out


    def compute_loss(self, out):
        rec_loss = self.base_rec.compute_loss(out)

        guide_loss = self.mse_loss(out['H_usr'], out['H_usr_sim']) + \
                     self.mse_loss(out['H_pos'], out['H_pos_sim']) + \
                     self.mse_loss(out['H_neg'], out['H_neg_sim'])

        return rec_loss + self.w_guide * guide_loss
