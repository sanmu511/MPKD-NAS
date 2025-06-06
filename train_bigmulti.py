import os
import numpy as np
import pandas
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from model_aux import Transformer_Auxiliary
from early_stop_tool import EarlyStopping
from data_loader import data_generator
from args import Config, Path
from caculate_metrics import Caculate_Metrics


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


def train(save_all_checkpoint=False):
    config = Config()
    path = Path()
    caculate_metrics = Caculate_Metrics(config,path,'bigmulti2_Kfold_models')
    # tem
    caculate_metrics._calc_metrics()
    selected_d = {"outs": [], "trg": []}

    dataset, labels, val_loader = data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)

    kf = StratifiedKFold(n_splits=config.num_fold, shuffle=True, random_state=0)

    fold_bestacc = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
        selected = 0.0
        print('\n', '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)
        # 设置保存模型的地址
        if not os.path.exists('./saved/bigmulti2_Kfold_models/fold{}'.format(fold)):
            os.makedirs('./saved/bigmulti2_Kfold_models/fold{}'.format(fold))
        result_path = './saved/bigmulti2_Kfold_models/fold{}'.format(fold)

        X_train, X_test = dataset[train_idx], dataset[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_set = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=False)

        model = Transformer_Auxiliary(config)
        ######################tem############################
        model.load_state_dict(torch.load('./saved/bigmulti_Kfold_models/fold8/model.pkl'))
        model = model.to(config.device)

        criterion = nn.CrossEntropyLoss()

        # AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

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
        for epoch in range(config.num_epochs):
            overall_outs = []
            overall_trgs = []
            running_loss = 0.0
            correct = 0

            model.train()

            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (data, target) in loop:
                data = data.to(config.device)
                target = target.to(config.device)
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                logits,ss_logits = model(data)

                # loss_cls = torch.tensor(0.).cuda()
                # loss_cls = loss_cls + criterion(output, target.long())  # 最终输出和target
                # loss = loss_cls
                loss_cls = torch.tensor(0.).cuda()
                for i in range(len(ss_logits)):
                    loss_cls = loss_cls + criterion(ss_logits[i], target.long())  # 每一个分类器的输出和label
                loss_cls = loss_cls + criterion(logits, target.long())  # 最终输出和target
                loss = loss_cls

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
                selected = val_acc #这里的val_acc是一个epoch的结果
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            # 当前epoch是最后一个epoch的时候才会产生overall_outs
            if epoch == (config.num_epochs)-1:
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
            #################################模型保存#############################################3
            # Check whether to continue training. If save_all_checkpoint=False, the model name will be ‘model.pkl'
            # 默认设置save_all_checkpoint=False，意思是保存一个fold中验证acc最高的模型
            early_stopping(val_acc, model, path='./saved/bigmulti2_Kfold_models/fold{}/model_{}_epoch{}.pkl'.format(fold, fold, epoch))

            if early_stopping.early_stop:
                print("Early stopping at epoch ", epoch)
                break
        # 记录每个fold的最佳准确率
        fold_bestacc.append(best_acc)
        ########################保存模型输出结果方便计算metrics##############################################
        outs_name = "outs_" + str(fold)
        trgs_name = "trgs_" + str(fold)
        np.save(result_path+'/'+outs_name, overall_outs) # 保存本fold的最佳epoch的模型输出
        np.save(result_path+'/'+trgs_name, overall_trgs)

        if fold == config.num_fold - 1: #最后一个fold
        ###########################计算性能指标并输出为excel##########################################
        # 是对所有fold的计算
            caculate_metrics._calc_metrics()

        # np.save('./Kfold_models/fold{}/train_LOSS.npy'.format(fold), np.array(train_LOSS))
        # np.save('./Kfold_models/fold{}/train_ACC.npy'.format(fold), np.array(train_ACC))
        # np.save('./Kfold_models/fold{}/test_LOSS.npy'.format(fold), np.array(test_LOSS))
        # np.save('./Kfold_models/fold{}/test_ACC.npy'.format(fold), np.array(test_ACC))
        # np.save('./Kfold_models/fold{}/val_LOSS.npy'.format(fold), np.array(val_LOSS))
        # np.save('./Kfold_models/fold{}/val_ACC.npy'.format(fold), np.array(val_ACC))
        if not os.path.exists('./result_excel/bigmulti2/fold{}'.format(fold)):
            os.makedirs('./result_excel/bigmulti2/fold{}'.format(fold))
        path = './result_excel/bigmulti2/fold{}/big_fold{}.xlsx'.format(fold, fold)
        write_to_excel(train_LOSS, train_ACC, test_LOSS, test_ACC, val_LOSS, val_ACC, path)

        del model
        # return train_ACC,train_LOSS,test_ACC,test_LOSS,val_ACC,val_LOSS
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
    if not os.path.exists('./saved/bigmulti2_Kfold_models/best_fold'):
        os.makedirs('./saved/bigmulti2_Kfold_models/best_fold')
    with open("./saved/bigmulti2_Kfold_models/best_fold/best_fold.txt", 'a') as f:
        f.write(str(fold_best))

