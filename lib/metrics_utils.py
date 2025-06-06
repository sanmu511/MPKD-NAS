import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def calculate_roc_auc(y_true, y_pred_proba, n_classes):
    """计算ROC曲线和AUC值"""
    # 将标签进行one-hot编码
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算每个类别的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和AUC值
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return fpr, tpr, roc_auc

def plot_roc_curves(fpr, tpr, roc_auc, n_classes, save_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    # 绘制每个类别的ROC曲线
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 .format(i, roc_auc[i]))
    
    # 绘制微平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {0:0.2f})'
             .format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def calculate_class_metrics(y_true, y_pred):
    """计算每个类别的精确率、召回率和F1分数"""
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    return precision, recall, f1

def calculate_distillation_metrics(student_logits, teacher_logits, temperature=1.0):
    """计算知识蒸馏相关指标"""
    # 计算KL散度
    student_probs = torch.nn.functional.softmax(student_logits / temperature, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    kl_div = torch.nn.functional.kl_div(
        torch.log(student_probs),
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # 计算JS散度
    mean_probs = (student_probs + teacher_probs) / 2
    js_div = 0.5 * (
        torch.nn.functional.kl_div(
            torch.log(student_probs),
            mean_probs,
            reduction='batchmean'
        ) +
        torch.nn.functional.kl_div(
            torch.log(teacher_probs),
            mean_probs,
            reduction='batchmean'
        )
    ) * (temperature ** 2)
    
    # 计算MSE损失
    mse_loss = torch.nn.functional.mse_loss(student_logits, teacher_logits)
    
    return {
        'kl_divergence': kl_div.item(),
        'js_divergence': js_div.item(),
        'mse_loss': mse_loss.item()
    }

def calculate_model_complexity(model):
    """计算模型的参数量和FLOPs"""
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }