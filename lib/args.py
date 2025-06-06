import torch
import os


class Config(object):
    """args in model and trainer"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_fold = 10  #10
        self.num_classes = 5
        self.num_epochs = 200   #200              # Because early stopping is used, this parameter can be relatively large
        self.batch_size = 64  #64
        self.pad_size = 29                  # time dimension of TF image
        self.learning_rate = 5e-2 #5e-6
        self.dropout = 0.1                  # dropout rate in transformer encoder
        self.dim_model = 128       #128         # frequency of TF image
        self.forward_hidden = 1024     #1024     # hidden units of transformer encoder
        self.fc_hidden = 1024         #1024      # hidden units of FC layers
        self.num_head = 8             #8
        self.num_encoder = 16         #16      # number of encoders in single-channel feature extraction block
        self.num_encoder_multi = 4      #4    # number of encoders in multi-channel feature fusion block
        ##################################################
        # 学生模型超网络配置
        self.s_forward_hidden = 64  # 1024
        self.s_fc_hidden = 64  # 1024
        self.s_num_head = 8  # 8
        self.s_num_encoder = 1  # 16
        self.s_num_encoder_multi = 1  # 4
        
        # 架构搜索配置
        # 为每个encoder层定义独立的搜索空间，使架构搜索更加精确
        self.search_space = {
            # encoder1的搜索空间配置
            'encoder1': {
                'dim_model': [32, 64, 96, 128],  # 模型维度搜索范围
                'num_heads': [4, 8],              # 注意力头数搜索范围
                'mlp_ratio': [2, 3, 4],           # MLP扩展比例搜索范围
                'layer_num': [1, 2, 3, 4]         # 层数搜索范围
            },
            # encoder2的搜索空间配置
            'encoder2': {
                'dim_model': [32, 64, 96, 128],
                'num_heads': [4, 8],
                'mlp_ratio': [2, 3, 4],
                'layer_num': [1, 2, 3, 4]
            },
            # encoder3的搜索空间配置
            'encoder3': {
                'dim_model': [32, 64, 96, 128],
                'num_heads': [4, 8],
                'mlp_ratio': [2, 3, 4],
                'layer_num': [1, 2, 3, 4]
            },
            # encoder_multi的搜索空间配置（用于多通道特征融合）
            'encoder_multi': {
                'dim_model': [32, 64, 96, 128],
                'num_heads': [4, 8],
                'mlp_ratio': [2, 3, 4],
                'layer_num': [1, 2, 3, 4]
            }
        }
        self.population_size = 50
        self.max_epochs = 20
        self.select_num = 10
        self.mutation_num = 25
        self.m_prob = 0.1
        self.crossover_num = 15
        #####################蒸馏温度##################################
        self.kd_T = 3
        #####################metrics save dir#################################
        self.metrics_dir = 'saved/'
        # 存储的文件名
        self.name = 'smallmulti_nokd'


class Path(object):
    """path of files in this project"""
    def __init__(self):
        self.path_PSG = '../dataset/data_sleep_edf_20/sleep-cassette'
        self.path_hypnogram = '../dataset/data_sleep_edf_20/Hypnogram'
        self.path_raw_data = 'data/data_sleep_edf_20/data_array/raw_data'
        self.path_labels = 'data/data_sleep_edf_20/data_array/raw_data/labels'
        self.path_TF = 'data/data_sleep_edf_20/data_array/TF_data'

        if not os.path.exists(self.path_hypnogram):
            os.makedirs(self.path_hypnogram)

        if not os.path.exists(self.path_raw_data):
            os.makedirs(self.path_raw_data)

        if not os.path.exists(self.path_TF):
            os.makedirs(self.path_TF)

