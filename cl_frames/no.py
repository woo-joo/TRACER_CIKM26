from llm_enhancers.llm_enhancer import LLM_Enhancer
from cl_frames.cl_frame import CL_Frame



class No(CL_Frame):
    def __init__(self, args, llm_enhancer: LLM_Enhancer):
        super().__init__(args, llm_enhancer)

        self.__load_prev__(args)
        self.llm_enhancer.__init_with_X__(args)
