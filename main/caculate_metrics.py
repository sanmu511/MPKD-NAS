import numpy as np
from tqdm import tqdm

from sklearn.metrics import recall_score, accuracy_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

#from model import Transformer
from data_loader import data_generator


class Caculate_Metrics:
    def __init__(self, config, path, name):

        self.config = config
        self.path = path
        self.name = name

    def _calc_metrics(self):
        from sklearn.metrics import classification_report
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import os
        from os import walk
        import warnings
        warnings.filterwarnings("ignore")

        n_folds = self.config.num_fold
        all_outs = []
        all_trgs = []

        outs_list = []
        trgs_list = []
        #############保存路径##############
        # save_dir = os.path.abspath(os.path.join(self.config.metrics_dir, os.pardir))
        save_dir = './saved/'+ self.name+'/fold1'
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if "outs" in file:
                     outs_list.append(os.path.join(root, file))
                if "trgs" in file:
                     trgs_list.append(os.path.join(root, file))
        # 所有fold的输出
        if 1: # len(outs_list)==n_folds
            for i in range(len(outs_list)):
                outs = np.load(outs_list[i])
                trgs = np.load(trgs_list[i])
                all_outs.extend(outs)
                all_trgs.extend(trgs)

        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_trgs, all_outs)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df["accuracy"] = accuracy_score(all_trgs, all_outs)
        df["f1"] = f1_score(all_trgs,all_outs, average='macro')
        df = df * 100
        file_name = self.config.name + "_classification_report.xlsx"
        report_Save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_Save_path)

        cm_file_name = self.config.name + "_confusion_matrix.torch"
        cm_Save_path = os.path.join(save_dir, cm_file_name)
        torch.save(cm, cm_Save_path)

