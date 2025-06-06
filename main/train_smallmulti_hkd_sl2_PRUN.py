import os
import numpy as np
import pandas
from tqdm import tqdm
import pickle

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from s_model_aux2_PRUN import s_Transformer_Auxiliary
from model_aux2 import Transformer_Auxiliary
from early_stop_tool import EarlyStopping
from data_loader import data_generator
from args import Config, Path
import torch.nn.functional as F
from caculate_metrics import Caculate_Metrics
import knowledge_distiller
from pruning_utils import ChannelPruner

class DistillKL(nn.Module): # 传统的KD
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

def set_random_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)        # CPU
    torch.cuda.manual_seed(seed)   # GPU


def test(model, test_loader, config):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    pred = []
    label = []

    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(config.device)
            target = target.to(config.device)
            data, target = Variable(data), Variable(target)

            output,_ = model(data)
            test_loss += criterion(output, target.long()).item()

            pred.extend(np.argmax(output.data.cpu().numpy(), axis=1))
            label.extend(target.data.cpu().numpy())

        accuracy = accuracy_score(label, pred, normalize=True, sample_weight=None)

    return accuracy, test_loss, pred, label


def train(save_all_checkpoint=False, pruning=True, pruning_ratio=0.3):
    config = Config()
    path = Path()
    caculate_metrics = Caculate_Metrics(config, path, 'student_hkd_sl2_PRUN_Kfold_models')
    # tem calculate
    caculate_metrics._calc_metrics()
    selected_d = {"outs": [], "trg": []}

    dataset, labels, val_loader = data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)

    kf = StratifiedKFold(n_splits=config.num_fold, shuffle=True, random_state=0)

    fold_bestacc = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
        selected = 0.0
        print('\n', '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)
        if not os.path.exists('./saved/student_hkd_sl2_PRUN_Kfold_models/fold{}'.format(fold)):
            os.makedirs('./saved/student_hkd_sl2_PRUN_Kfold_models/fold{}'.format(fold))
        result_path = './saved/student_hkd_sl2_PRUN_Kfold_models/fold{}'.format(fold)

        X_train, X_test = dataset[train_idx], dataset[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_set = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=False)
        #########################设置学生和老师#####################################
        model = s_Transformer_Auxiliary(config)
        model = model.to(config.device)
        
        # 创建剪枝器并为每个encoder层设置相同的剪枝率
        pruner = None
        if pruning:
            pruner = ChannelPruner()
            # 为三个encoder层设置相同的剪枝率
            base_ratio = pruning_ratio
            pruner.register_layer('backbone.transformer_encoder_1', model.backbone.transformer_encoder_1, base_ratio)  # 所有层使用相同剪枝率
            pruner.register_layer('backbone.transformer_encoder_2', model.backbone.transformer_encoder_2, base_ratio)  # 所有层使用相同剪枝率
            pruner.register_layer('backbone.transformer_encoder_3', model.backbone.transformer_encoder_3, base_ratio)  # 所有层使用相同剪枝率

        # 读取best_fold对应的teacher model
        with open('./saved/bigmulti2_Kfold_models/best_fold/best_fold.txt', 'r', encoding='utf-8') as f:
            best_fold = f.readline()
        tmodel = Transformer_Auxiliary(config)
        tmodel.load_state_dict(
            torch.load('./saved/bigmulti2_Kfold_models/fold{}/model.pkl'.format(int(best_fold))))
        
        # 蒸馏网络
        dnet = knowledge_distiller.WSLDistiller(tmodel, model)
        dnet = dnet.to(config.device)

        criterion = nn.CrossEntropyLoss()
        criterion_div = DistillKL(config.kd_T)

        # AdamW optimizer
        optimizer = optim.AdamW(dnet.parameters(), lr=config.learning_rate, weight_decay=0.01)

        # apply early_stop. If you want to view the full training process, set the save_all_checkpoint True
        early_stopping = EarlyStopping(patience=20, verbose=True, save_all_checkpoint=save_all_checkpoint)

        # evaluating indicator
        train_ACC = []
        train_LOSS = []
        test_ACC = []
        test_LOSS = []
        val_ACC = []
        val_LOSS = []

        best_acc, best_epoch = 0, 0
        
        # 剪枝训练分为两个阶段：预训练和剪枝后微调
        pruning_stage = False  # 标记当前是否处于剪枝后微调阶段
        for epoch in range(config.num_epochs):
            overall_outs = []
            overall_trgs = []
            running_loss = 0.0
            correct = 0

            dnet.train()
            dnet.s_net.train()
            dnet.t_net.train()

            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (data, target) in loop:
                data = data.to(config.device)
                target = target.to(config.device)
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                logits, loss = dnet(data, target)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                train_acc_batch = np.sum(np.argmax(np.array(logits.data.cpu()), axis=1) == np.array(target.data.cpu())) / (target.shape[0])
                loop.set_postfix(train_acc=train_acc_batch, loss=loss.item())
                correct += np.sum(np.argmax(np.array(logits.data.cpu()), axis=1) == np.array(target.data.cpu()))

            train_acc = correct / len(train_loader.dataset)
            ############################################################
            test_acc, test_loss, _, __ = test(model, test_loader, config)
            val_acc, val_loss, outs, trgs = test(model, val_loader, config)
            ##########################一个fold只存储最佳epoch的结果############
            if val_acc > selected:
                selected = val_acc  # 这里的val_acc是一个epoch的结果
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            # 当前epoch是最后一个epoch的时候才会产生overall_outs
            if epoch == (config.num_epochs) - 1:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])
            # 记录本fold的最佳val_acc
            if val_acc > best_acc:
                best_acc = val_acc
            print('Epoch: ', epoch,
                  '| train loss: %.4f' % running_loss, '| train acc: %.4f' % train_acc,
                  '| val acc: %.4f' % val_acc, '| val loss: %.4f' % val_loss,
                  '| test acc: %.4f' % test_acc, '| test loss: %.4f' % test_loss)

            train_ACC.append(train_acc)
            train_LOSS.append(running_loss)
            test_ACC.append(test_acc)
            test_LOSS.append(test_loss)
            val_ACC.append(val_acc)
            val_LOSS.append(val_loss)

            # 渐进式剪枝实现
            if pruning and epoch >= 30:  # 从第30个epoch开始应用渐进式剪枝
                # 计算当前的剪枝进度
                pruning_progress = min(1.0, (epoch - 30) / 20)  # 在20个epoch内逐步增加剪枝率
                
                # 计算当前epoch的实际剪枝率
                current_ratio = pruning_ratio * pruning_progress
                
                # 更新每个encoder层的剪枝率 - 使用相同的剪枝率
                pruner.pruning_ratios['backbone.transformer_encoder_1'] = current_ratio
                pruner.pruning_ratios['backbone.transformer_encoder_2'] = current_ratio
                pruner.pruning_ratios['backbone.transformer_encoder_3'] = current_ratio
                
                # 计算并应用剪枝
                if not pruning_stage:
                    print("\nStarting progressive pruning at epoch", epoch)
                    pruning_stage = True
                
                # 计算通道重要性并生成剪枝掩码
                pruner.compute_importance(model, train_loader, config.device)
                pruner.generate_masks()
                
                # 应用剪枝
                pruner.apply_pruning(model)
                
                # 监控剪枝效果
                pruned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Current pruning ratio: {current_ratio:.3f}")
                print(f"Remaining parameters: {pruned_params:,}")
                
                # 如果达到最佳验证准确率，保存剪枝后的模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    # 保存剪枝后的模型和剪枝配置
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'pruning_masks': pruner.masks,
                        'pruning_ratios': pruner.pruning_ratios,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch
                    }, os.path.join(result_path, 'best_pruned_model.pth'))
                
                # 重置优化器学习率进行微调
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.learning_rate * 0.1
                
                print("Pruning applied, continuing fine-tuning...")
                continue
            
            # Check whether to continue training. If save_all_checkpoint=False, the model name will be 'model.pkl'
            model_save_path = './saved/student_hkd_sl2_PRUN_Kfold_models/fold{}/model_{}_epoch{}.pkl'.format(fold, fold, epoch)
            early_stopping(val_acc, model, path=model_save_path)

            if early_stopping.early_stop:
                print("Early stopping at epoch ", epoch)
                break
        # 记录每个fold的最佳准确率
        fold_bestacc.append(best_acc)
        ########################保存模型输出结果方便计算metrics##############################################
        outs_name = "outs_" + str(fold)
        trgs_name = "trgs_" + str(fold)
        np.save(result_path + '/' + outs_name, overall_outs)  # 保存本fold的最佳epoch的模型输出
        np.save(result_path + '/' + trgs_name, overall_trgs)

        if fold == config.num_fold - 1:  # 最后一个fold
        ###########################计算性能指标并输出为excel##########################################
        # 是对所有fold的计算
            caculate_metrics._calc_metrics()

        # np.save('./Kfold_models/fold{}/train_LOSS.npy'.format(fold), np.array(train_LOSS))
        # np.save('./Kfold_models/fold{}/train_ACC.npy'.format(fold), np.array(train_ACC))
        # np.save('./Kfold_models/fold{}/test_LOSS.npy'.format(fold), np.array(test_LOSS))
        # np.save('./Kfold_models/fold{}/test_ACC.npy'.format(fold), np.array(test_ACC))
        # np.save('./Kfold_models/fold{}/val_LOSS.npy'.format(fold), np.array(val_LOSS))
        # np.save('./Kfold_models/fold{}/val_ACC.npy'.format(fold), np.array(val_ACC))
        if not os.path.exists('./result_excel/smallmulti_hkd_sl2/fold{}'.format(fold)):
            os.makedirs('./result_excel/smallmulti_hkd_sl2/fold{}'.format(fold))
        path = './result_excel/smallmulti_hkd_sl2/fold{}/smodel_hkd_sl_fold{}.xlsx'.format(fold, fold)
        write_to_excel(train_LOSS, train_ACC, test_LOSS, test_ACC, val_LOSS, val_ACC, path)

        del model
        #return train_ACC,train_LOSS,test_ACC,test_LOSS,val_ACC,val_LOSS
    # 最佳fold
    fold_best = fold_bestacc.index(max(fold_bestacc))
    return fold_best

def write_to_excel(loss,acc,te_loss,te_acc,val_loss,val_acc,path):
    result_excel = pandas.DataFrame()
    result_excel["tr_loss"] = loss
    result_excel["tr_acc"]=acc
    result_excel["te_loss"] = te_loss
    result_excel["te_acc"] = te_acc
    result_excel["val_loss"] = val_loss
    result_excel["val_acc"] = val_acc
    # 写入excel
    result_excel.to_excel(path)

if __name__ == '__main__':
    set_random_seed(0)
    #train_ACC, train_LOSS, test_ACC, test_LOSS, val_ACC, val_LOSS = [],[],[],[],[],[]
    fold_best = train(save_all_checkpoint=False)
    if not os.path.exists('./saved/student_hkd_sl2_Kfold_models/best_fold'):
        os.makedirs('./saved/student_hkd_sl2_Kfold_models/best_fold')
    with open("./saved/student_hkd_sl2_Kfold_models/best_fold/best_fold.txt", 'a') as f:
        f.write(str(fold_best))

