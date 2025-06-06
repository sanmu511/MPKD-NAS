import torch
import torch.nn as nn
import numpy as np

class ChannelPruner:
    """
    基于重要性的通道剪枝工具类
    用于对Transformer模型的各层进行结构化剪枝
    """
    def __init__(self):
        self.importance_scores = {}
        self.masks = {}
        self.pruning_ratios = {}
    
    def register_layer(self, layer_name, module, pruning_ratio=0.3):
        """
        注册需要剪枝的层
        
        Args:
            layer_name: 层的名称
            module: 需要剪枝的模块
            pruning_ratio: 剪枝比例，表示要剪掉的通道比例
        """
        self.pruning_ratios[layer_name] = pruning_ratio
    
    def compute_importance(self, model, data_loader, device):
        """
        计算各层通道的重要性分数
        
        Args:
            model: 需要剪枝的模型
            data_loader: 用于评估重要性的数据加载器
            device: 计算设备
        """
        # 注册钩子函数来收集特征图
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 为每个注册的层添加前向钩子
        hooks = []
        for name, module in model.named_modules():
            if name in self.pruning_ratios:
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        # 在数据上运行模型以收集激活值
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(device)
                model(data)
                if batch_idx >= 10:  # 限制使用的批次数量
                    break
        
        # 计算每个通道的重要性分数
        for name in self.pruning_ratios.keys():
            if name in activation:
                act = activation[name]
                if len(act.shape) == 4:  # 卷积层输出
                    importance = torch.mean(torch.abs(act), dim=[0, 2, 3])
                else:  # Transformer层输出
                    importance = torch.mean(torch.abs(act), dim=[0, 1])
                self.importance_scores[name] = importance.cpu().numpy()
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
    
    def generate_masks(self):
        """
        基于重要性分数生成剪枝掩码
        """
        for name, importance in self.importance_scores.items():
            pruning_ratio = self.pruning_ratios[name]
            num_channels = len(importance)
            num_to_prune = int(num_channels * pruning_ratio)
            
            # 找到重要性最低的通道索引
            indices = np.argsort(importance)
            prune_indices = indices[:num_to_prune]
            keep_indices = indices[num_to_prune:]
            
            # 创建二进制掩码，0表示剪枝，1表示保留
            mask = torch.ones(num_channels)
            mask[prune_indices] = 0
            
            self.masks[name] = mask
    
    def apply_pruning(self, model):
        """
        应用剪枝掩码到模型
        
        Args:
            model: 需要剪枝的模型
        """
        for name, module in model.named_modules():
            if name in self.masks:
                mask = self.masks[name].to(next(module.parameters()).device)
                
                # 对不同类型的层应用不同的剪枝方法
                if isinstance(module, nn.Linear):
                    module.weight.data *= mask.view(1, -1)  # 对输出维度剪枝
                elif isinstance(module, nn.TransformerEncoderLayer) or hasattr(module, 'attn'):
                    # 对注意力层的权重进行剪枝
                    if hasattr(module, 'attn') and hasattr(module.attn, 'qkv'):
                        # 对QKV投影进行剪枝
                        head_dim = module.attn.head_dim
                        num_heads = module.attn.num_heads
                        for i in range(num_heads):
                            if mask[i] == 0:  # 如果这个头需要被剪枝
                                start_idx = i * head_dim
                                end_idx = (i + 1) * head_dim
                                module.attn.qkv.weight.data[:, start_idx:end_idx] = 0
                                if module.attn.qkv.bias is not None:
                                    module.attn.qkv.bias.data[start_idx:end_idx] = 0
    
    def get_pruned_model(self, model):
        """
        获取剪枝后的模型
        
        Args:
            model: 原始模型
            
        Returns:
            剪枝后的模型
        """
        # 首先应用掩码
        self.apply_pruning(model)
        return model

# 特定于Transformer模型的剪枝函数
def prune_transformer_encoder(encoder, pruning_ratio=0.3):
    """
    对Transformer编码器层进行剪枝
    
    Args:
        encoder: TransformerEncoder模块
        pruning_ratio: 剪枝比例
        
    Returns:
        剪枝后的编码器
    """
    # 计算每个注意力头的重要性
    importance_scores = []
    
    # 这里简化处理，实际应用中应该基于数据计算重要性
    for i, layer in enumerate(encoder.layers):
        if hasattr(layer, 'attn'):
            # 假设注意力层有权重可以用来评估重要性
            if hasattr(layer.attn, 'qkv'):
                weight = layer.attn.qkv.weight.data
                # 按注意力头分组计算权重范数
                head_dim = layer.attn.head_dim
                num_heads = layer.attn.num_heads
                head_importance = []
                for h in range(num_heads):
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    head_weight = weight[:, start_idx:end_idx]
                    importance = torch.norm(head_weight).item()
                    head_importance.append(importance)
                importance_scores.append(head_importance)
    
    # 如果没有计算出重要性分数，返回原始编码器
    if not importance_scores:
        return encoder
    
    # 确定要剪枝的头
    all_heads = np.concatenate(importance_scores)
    threshold = np.percentile(all_heads, pruning_ratio * 100)
    
    # 应用剪枝
    for i, layer in enumerate(encoder.layers):
        if hasattr(layer, 'attn') and i < len(importance_scores):
            head_importance = importance_scores[i]
            for h, imp in enumerate(head_importance):
                if imp <= threshold:  # 剪枝不重要的头
                    # 将对应头的权重置零
                    if hasattr(layer.attn, 'qkv'):
                        head_dim = layer.attn.head_dim
                        start_idx = h * head_dim
                        end_idx = (h + 1) * head_dim
                        layer.attn.qkv.weight.data[:, start_idx:end_idx] = 0
                        if layer.attn.qkv.bias is not None:
                            layer.attn.qkv.bias.data[start_idx:end_idx] = 0
    
    return encoder