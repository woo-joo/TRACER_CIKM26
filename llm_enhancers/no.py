from base_rec import Base_Recommender
from llm_enhancers.llm_enhancer import LLM_Enhancer



class No(LLM_Enhancer):
    def __init__(self, args, base_rec: Base_Recommender):
        super().__init__(args, base_rec)
