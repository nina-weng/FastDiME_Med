import os
from numpy import dot
from numpy.linalg import norm

def get_ckpt(dict_model_list):
    for k in dict_model_list.keys():
        pth = dict_model_list[k]
        pth = pth+'/checkpoints/'
        ckpt_files = os.listdir(pth)
        ckpt_files.sort()
        pth = pth+ckpt_files[-1]
        dict_model_list[k] = pth
    return dict_model_list


def cos_similarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim