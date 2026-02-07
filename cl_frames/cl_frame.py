import torch
from torch import nn

from llm_enhancers.llm_enhancer import LLM_Enhancer



class CL_Frame(nn.Module):
    def __init__(self, args, llm_enhancer: LLM_Enhancer):
        super().__init__()

        self.llm_enhancer = llm_enhancer


    def __load_prev__(self, args):
        if args.cl_frame == 'Full_Batch': return

        idx = int(args.dataset[-1])
        frame = 'Full_Batch' if idx == 1 else 'PISA' if args.cl_frame == 'PISA_F' else args.cl_frame
        weight_path = f'weights/{args.base_rec}-{args.llm_enhancer}/{frame}/{args.dataset[:-1]}{idx-1}.pt'
        state_dict = torch.load(weight_path)
        self.load_state_dict(state_dict)


    def forward(self, usr, pos, neg): return self.llm_enhancer(usr, pos, neg)
    def predict(self, usr): return self.llm_enhancer.predict(usr)
    def compute_loss(self, out): return self.llm_enhancer.compute_loss(out)
