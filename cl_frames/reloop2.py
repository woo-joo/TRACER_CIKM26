import torch
from torch import nn

from llm_enhancers.llm_enhancer import LLM_Enhancer
from cl_frames.cl_frame import CL_Frame



class ReLoop2(CL_Frame):
    def __init__(self, args, llm_enhancer: LLM_Enhancer):
        super().__init__(args, llm_enhancer)

        self.b = 8
        self.k = args.k
        self.lambd = args.lambd
        self.num_item_all = args.num_item_all

        self.softmax = nn.Softmax(dim=-1)

        self.__load_prev__(args)
        self.__load_error_memory__(args)
        self.llm_enhancer.__init_with_X__(args)


    def __load_error_memory__(self, args):
        idx = int(args.dataset[-1])
        if idx == 1:
            self.hasher = torch.randn(args.k, self.b, args.d).to(args.device)
            self.error_memory_x = torch.zeros(args.k, 2 ** self.b).to(args.device)
            self.error_memory_y = torch.zeros(args.k, 2 ** self.b, args.num_item_all).to(args.device)
            return

        error_path = f'weights/{args.base_rec}-{args.llm_enhancer}/ReLoop2/{args.dataset[:-1]}{idx-1}_error.pt'
        error_memory = torch.load(error_path)
        self.hasher = error_memory['hasher'].to(args.device)
        self.error_memory_x = error_memory['x'].to(args.device)
        self.error_memory_y = error_memory['y'].to(args.device)


    def __get_hash_index__(self, query):
        index_bool = torch.permute(self.hasher @ query.T, (2, 0, 1)) > 0
        power2 = 2 ** torch.arange(self.b).to(index_bool.device)
        index = torch.sum(index_bool * power2, dim=-1)

        return index.T


    def forward(self, usr, pos, neg):
        out = self.llm_enhancer(usr, pos, neg)

        with torch.no_grad():
            index = self.__get_hash_index__(out['H_usr'])
            x = torch.ones_like(index).float()
            self.error_memory_x.scatter_add_(dim=1, index=index, src=x)

            index = index[..., None].expand(-1, -1, self.num_item_all)
            y = nn.functional.one_hot(pos, num_classes=self.num_item_all).float()
            y = y[None, ...].expand(self.k, -1, -1)
            self.error_memory_y.scatter_add_(dim=1, index=index, src=y)

        return out


    def __compute_y_bar__(self, user_emb):
        index = self.__get_hash_index__(user_emb)
        norm = torch.sum(self.error_memory_x, dim=1, keepdim=True)
        s = torch.gather(self.error_memory_x, dim=1, index=index) / norm

        index = index[..., None].expand(-1, -1, self.num_item_all)
        norm = norm[..., None]
        y = torch.gather(self.error_memory_y, dim=1, index=index) / norm

        y_bar = torch.sum(self.softmax(s)[..., None] * y, dim=0)

        return y_bar


    def predict(self, usr):
        y_base = self.softmax(self.llm_enhancer.predict(usr))

        user_emb = self.llm_enhancer.H_user[usr]
        y_bar = self.__compute_y_bar__(user_emb)

        return (1 - self.lambd) * y_base + self.lambd * y_bar
