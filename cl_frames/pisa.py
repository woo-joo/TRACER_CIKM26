import torch
from torch import nn

from llm_enhancers.llm_enhancer import LLM_Enhancer
from cl_frames.cl_frame import CL_Frame



class PISA(CL_Frame):
    def __init__(self, args, llm_enhancer: LLM_Enhancer):
        super().__init__(args, llm_enhancer)

        self.batch_size = args.batch_size
        self.num_user_cur = args.num_user_cur
        self.num_item_cur = args.num_item_cur
        self.r = args.r
        self.w_cl = args.w_cl
        self.adj = {user: items for user, (items, _, _) in args.data.items()}

        self.softmax = nn.Softmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        self.__load_knowledge__(args)
        self.__load_prev__(args)
        self.llm_enhancer.__init_with_X__(args)


    def __kmeans__(self, X, K=20, max_iters=100, tolerance=1e-4):
        centroids = X[torch.randperm(X.shape[0])[:K].to(X.device)]
        for _ in range(max_iters):
            distances = torch.cdist(X, centroids)
            assignments = torch.argmin(distances, dim=1)

            assignments_one_hot = nn.functional.one_hot(assignments, num_classes=K).float()
            counts = assignments_one_hot.sum(dim=0)
            sums = assignments_one_hot.T @ X
            new_centroids = sums / (counts[:, None] + 1e-8)

            empty_mask = counts == 0
            num_empty = torch.sum(empty_mask).item()
            if num_empty > 0:
                min_dist = torch.min(distances, dim=1).values
                new_centroids[empty_mask] = X[torch.topk(min_dist, k=num_empty).indices]

            if num_empty == 0 and torch.all(torch.abs(new_centroids - centroids) < tolerance): break

        return centroids


    @torch.no_grad()
    def __load_knowledge__(self, args):
        self.to(args.device)

        dir_name = f'weights/{args.base_rec}-{args.llm_enhancer}/PISA'
        idx = int(args.dataset[-1])

        dir_name_back = dir_name.replace('PISA', 'Full_Batch') if idx == 1 else dir_name
        weight_path_back = f'{dir_name_back}/{args.dataset[:-1]}{idx-1}.pt'
        state_dict_back = torch.load(weight_path_back)
        self.load_state_dict(state_dict_back)
        self.llm_enhancer.update_H()
        self.H_user_back = self.llm_enhancer.H_user.detach().clone()
        self.H_item_back = self.llm_enhancer.H_item.detach().clone()

        dir_name_fore = dir_name.replace('PISA', 'PISA_F')
        weight_path_fore = f'{dir_name_fore}/{args.dataset}.pt'
        state_dict_fore = torch.load(weight_path_fore)
        self.load_state_dict(state_dict_fore)
        self.llm_enhancer.update_H()
        self.H_user_fore = self.llm_enhancer.H_user.detach().clone()
        self.H_item_fore = self.llm_enhancer.H_item.detach().clone()

        self.to('cpu')

        c = self.__kmeans__(self.H_item_back[:args.num_item_pre])
        q = torch.zeros(self.num_user_cur, c.shape[0]).to(args.device)
        q[:args.num_user_pre] = self.H_user_back[:args.num_user_pre] @ c.T
        self.q = self.softmax(q)


    def __compute_jsd__(self, p, q):
        m = (p + q) / 2

        kl_p_m = torch.sum(nn.functional.kl_div(torch.log(m), p, reduction='none'), dim=-1)
        kl_q_m = torch.sum(nn.functional.kl_div(torch.log(m), q, reduction='none'), dim=-1)

        jsd = (kl_p_m + kl_q_m) / 2
        
        return jsd


    @torch.no_grad()
    def update_preference(self):
        self.llm_enhancer.update_H()

        c = self.__kmeans__(self.llm_enhancer.H_item[:self.num_item_cur])
        p = self.llm_enhancer.H_user[:self.num_user_cur] @ c.T
        p = self.softmax(p)

        pref_shift = self.__compute_jsd__(p, self.q)
        w_p = torch.sigmoid(pref_shift - torch.mean(pref_shift))
        w_s = 1 - w_p

        self.w_s = torch.where(w_s >= torch.quantile(w_s, 1 - self.r), w_s, 0.0)
        self.w_p = torch.where(w_p >= torch.quantile(w_p, 1 - self.r), w_p, 0.0)


    def forward(self, usr, pos, neg):
        out = self.llm_enhancer(usr, pos, neg)

        item, tag = [], []
        for idx, u in enumerate(usr):
            items = self.adj.get(u.item(), [])
            item += items
            tag += [idx] * len(items)
        item = torch.tensor(item).to(usr.device)
        tag = torch.tensor(tag).to(usr.device)

        out['user'] = usr
        out['item'] = item
        out['tag'] = tag

        return out


    def __compute_info_nce_loss__(self, idx, z_i, z_j, weight, tag=None):
        mask = (idx[:, None] == idx[None, :])
        mask.fill_diagonal_(0)
        if tag is not None:
            weight = weight[tag]
            mask = mask[tag]

        z_i = nn.functional.normalize(z_i)
        z_j = nn.functional.normalize(z_j)

        logit = z_i @ z_j.T
        logit[mask] = -10000.0
        if tag is not None: label = tag
        else: label = torch.arange(logit.shape[0]).to(logit.device)

        info_nce_loss = weight * self.ce_loss(logit, label)

        return info_nce_loss


    def compute_loss(self, out):
        user, item, tag = out['user'], out['item'], out['tag']
        H_user = out['H_usr']

        rec_loss = self.llm_enhancer.compute_loss(out)

        H_user_back = self.H_user_back[user]
        H_item_back = self.H_item_back[item]
        w_s = self.w_s[user]
        se_s_loss = self.__compute_info_nce_loss__(user, H_user_back, H_user, w_s)
        se_n_loss = []
        for i in range(0, item.shape[0], self.batch_size):
            j = i + self.batch_size
            se_n_loss.append(self.__compute_info_nce_loss__(user, H_item_back[i:j], H_user, w_s, tag[i:j]))
        se_n_loss = torch.cat(se_n_loss)
        se_loss = torch.mean(se_s_loss) + torch.mean(se_n_loss)
        
        H_user_fore = self.H_user_fore[user]
        H_item_fore = self.H_item_fore[item]
        w_p = self.w_p[user]
        pe_s_loss = self.__compute_info_nce_loss__(user, H_user_fore, H_user, w_p)
        pe_n_loss = []
        for i in range(0, item.shape[0], self.batch_size):
            j = i + self.batch_size
            pe_n_loss.append(self.__compute_info_nce_loss__(user, H_item_fore[i:j], H_user, w_p, tag[i:j]))
        pe_n_loss = torch.cat(pe_n_loss)
        pe_loss = torch.mean(pe_s_loss) + torch.mean(pe_n_loss)

        return rec_loss + self.w_cl * (se_loss + pe_loss)
