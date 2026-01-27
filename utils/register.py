from llm_encoders import *

def register_llm_model():
    llm_model_dict = {}
    llm_model_dict['rnafm'] = RnaFMEmbeddingExtractor
    llm_model_dict['RiNALMo'] = RiNALMoEmbeddingExtractor
    return llm_model_dict