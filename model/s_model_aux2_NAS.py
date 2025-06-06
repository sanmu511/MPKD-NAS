import math

import torch
from torch import nn
from torch.autograd import Variable
from model.supernet_transformer import TransformerEncoderLayer
from model.supernet_transformer import TransformerEncoder
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.utils import trunc_normal_

class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model=128, dropout=0.2, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe:[1, 30, 128]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        # 超网络配置参数
        self.super_dim_model = config.dim_model
        self.super_num_heads = config.s_num_head
        self.super_mlp_ratio = config.s_forward_hidden / config.dim_model
        self.super_dropout = config.dropout
        self.super_layer_num = config.s_num_encoder

        # 采样配置参数
        self.sample_dim_model = None
        self.sample_num_heads = None
        self.sample_mlp_ratio = None
        self.sample_dropout = None
        self.sample_layer_num = None

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        # 使用可搜索的TransformerEncoderLayer
        encoder_layer = TransformerEncoderLayer(
            dim=config.dim_model,
            num_heads=config.s_num_head,
            mlp_ratio=config.s_forward_hidden / config.dim_model,
            dropout=config.dropout,
            attn_drop=config.dropout
        )
        self.transformer_encoder_1 = TransformerEncoder(encoder_layer, num_layers=config.s_num_encoder)
        self.transformer_encoder_2 = TransformerEncoder(encoder_layer, num_layers=config.s_num_encoder)
        self.transformer_encoder_3 = TransformerEncoder(encoder_layer, num_layers=config.s_num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = LayerNormSuper(super_embed_dim=config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)
        encoder_layer_multi = TransformerEncoderLayer(
            dim=config.dim_model * 3,
            num_heads=config.s_num_head,
            mlp_ratio=config.s_forward_hidden / config.dim_model,
            dropout=config.dropout,
            attn_drop=config.dropout
        )
        self.transformer_encoder_multi = TransformerEncoder(encoder_layer_multi, num_layers=config.s_num_encoder_multi)

        # 使用可搜索的Linear层
        self.fc1 = nn.Sequential(
            LinearSuper(config.pad_size * config.dim_model * 3, config.s_fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            LinearSuper(config.s_fc_hidden, config.num_classes)
        )

    def set_sample_config(self, config: dict):
        self.sample_dim_model = config['dim_model']
        self.sample_num_heads = config['num_heads']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_dim_model, self.super_dim_model)
        
        # 设置各个模块的采样配置
        self.layer_norm.set_sample_config(self.sample_dim_model * 3)
        for encoder in [self.transformer_encoder_1, self.transformer_encoder_2, self.transformer_encoder_3]:
            encoder.set_sample_config(self.sample_dim_model, self.sample_num_heads, self.sample_mlp_ratio)
        self.transformer_encoder_multi.set_sample_config(self.sample_dim_model * 3, self.sample_num_heads, self.sample_mlp_ratio)
        
        # 设置全连接层的采样配置
        self.fc1[0].set_sample_config(self.sample_dim_model * 3 * config.pad_size, config.s_fc_hidden)
        self.fc2[0].set_sample_config(config.s_fc_hidden, config.num_classes)

    def forward(self, x, is_feat=False):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x1 = self.position_single(x1)
        x2 = self.position_single(x2)
        x3 = self.position_single(x3)

        x1 = self.transformer_encoder_1(x1)     # (batch_size, 29, 128)
        x2 = self.transformer_encoder_2(x2)
        x3 = self.transformer_encoder_3(x3)

        x = torch.cat([x1, x2, x3], dim=2)
        f1 = x

        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        x = self.position_multi(x)
        f2 = x
        x = self.transformer_encoder_multi(x)
        f3 = x

        x = self.layer_norm(x + residual)  # residual connection

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        if is_feat:
            return x, [f1, f2, f3], residual
        else:
            return x


class Auxiliary_Classifier(nn.Module):
    def __init__(self, config):
        super(Auxiliary_Classifier, self).__init__()
        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)

        mlp_ratio = config.s_forward_hidden / config.dim_model,
        if isinstance(mlp_ratio, tuple):
            mlp_ratio= mlp_ratio[0]
        encoder_layer_multi = TransformerEncoderLayer(dim=config.dim_model * 3, num_heads=config.s_num_head,
                                                      mlp_ratio = mlp_ratio, dropout=config.dropout)
        self.transformer_encoder_multi = TransformerEncoder(encoder_layer_multi, num_layers=config.s_num_encoder_multi)

        self.fc1 = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model * 3, config.s_fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.s_fc_hidden, config.num_classes)
        )
        self.extractor1 = nn.Sequential(*[self.drop,self.layer_norm,self.position_multi,self.transformer_encoder_multi,self.layer_norm])
        self.extractor2 = nn.Sequential(*[self.transformer_encoder_multi,self.layer_norm])
        self.extractor3 = nn.Sequential(*[self.layer_norm])

    def forward(self, x, residual):
        ss_logits = []
        for i in range(len(x)):
            idx = i + 1
            extractor_layers = list(getattr(self, 'extractor' + str(idx)).children())
            if idx == 1:
                for j in range(0, 2):
                    x[i] = extractor_layers[j](x[i])
                residual_1 = x[i]
                for k in range(2, 4):
                    x[i] = extractor_layers[k](x[i])
                out = extractor_layers[-1](x[i] + residual_1)  # 最后一层的输出结果
                out = out.view(out.size(0), -1)
                out = self.fc1(out)
                out = self.fc2(out)
                ss_logits.append(out)
            else:
                for layer in extractor_layers[:-1]:  # 排除最后一层
                    x[i] = layer(x[i])
                out = extractor_layers[-1](x[i] + residual)  # 最后一层的输出结果
                out = out.view(out.size(0), -1)
                out = self.fc1(out)
                out = self.fc2(out)
                ss_logits.append(out)
        return ss_logits


class s_Transformer_Auxiliary(nn.Module):
    def __init__(self, config, drop_rate=0., embed_dim=768, img_size=224, patch_size=16, in_chans=3, pre_norm=True, num_classes=1000):
        super(s_Transformer_Auxiliary, self).__init__()
        self.backbone = Transformer(config)
        self.auxiliary_classifier = Auxiliary_Classifier(config)
        self.super_dropout = drop_rate
        self.super_embed_dim = embed_dim
        self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size,
                                                 in_chans=in_chans, embed_dim=embed_dim)
        self.blocks = nn.ModuleList()
        self.pre_norm = pre_norm
        self.num_classes = num_classes

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)
            self.norm.set_sample_config(sample_embed_dim=64)
        # classifier head
        self.head = LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, grad=False):
        logit,feats,residual = self.backbone(x, is_feat=True)
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_logits = self.auxiliary_classifier(feats,residual)
        return logit, ss_logits

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                blocks.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim[i],
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=sample_dropout,
                                        sample_out_dim=self.sample_output_dim[i],
                                        sample_attn_dropout=sample_attn_dropout)
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim