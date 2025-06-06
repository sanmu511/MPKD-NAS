import pickle
from args import Config, Path
from model_aux import Transformer_Auxiliary

import torch

with open('./saved/teacher_Kfold_models/best_fold/best_fold.txt', 'r', encoding='utf-8') as f:
    best_fold = f.readline()
config = Config()
tmodel = Transformer_Auxiliary(config)
tmodel = tmodel.load_state_dict(torch.load('./saved/teacher_Kfold_models/fold{}/model.pkl'.format(int(best_fold))))
print("a")