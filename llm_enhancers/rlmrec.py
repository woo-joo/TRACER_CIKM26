import torch
from torch import nn

from base_rec import Base_Recommender
from llm_enhancers.llm_enhancer import LLM_Enhancer



class RLMRec(LLM_Enhancer):
    def __init__(self, args, base_rec: Base_Recommender):
        super().__init__(args, base_rec)

        self.w_guide = args.w_sync

        self.adapter = nn.Sequential(nn.Linear(args.d_llm, (args.d_llm + args.d) // 2),
                                     nn.LeakyReLU(),
                                     nn.Linear((args.d_llm + args.d) // 2, args.d))

        self.ce_loss = nn.CrossEntropyLoss()

        nn.init.xavier_uniform_(self.adapter[0].weight)
        nn.init.xavier_uniform_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].bias)


    def forward(self, usr, pos, neg):
        out = self.base_rec(usr, pos, neg)
        
        out['usr'] = usr
        out['pos'] = pos
        out['neg'] = neg

        out['S_usr'] = self.adapter(self.X_user[usr])
        out['S_pos'] = self.adapter(self.X_item[pos])
        out['S_neg'] = self.adapter(self.X_item[neg])

        return out


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
        rec_loss = self.base_rec.compute_loss(out)

        guide_loss = self.__compute_info_nce_loss__(out['usr'], out['H_usr'], out['S_usr']) + \
                     self.__compute_info_nce_loss__(out['pos'], out['H_pos'], out['S_pos']) + \
                     self.__compute_info_nce_loss__(out['neg'], out['H_neg'], out['S_neg'])

        return rec_loss + self.w_guide * guide_loss
