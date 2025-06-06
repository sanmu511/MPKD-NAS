# MPKD-NAS
### Multi-Phase Knowledge Distillation Framework based on Neural Architecture Search
## Abstract
> Recently, the scaling of deep learning models intensified the conflict between deployment efficiency and performance, heightening the need for effective model lightweight. While Knowledge Distillation (KD) offers significant model compression benefits, it faces limitations such as single-stage distillation constraints and suboptimal performance under fixed model architectures. Neural Architecture Search (NAS) provides architectural optimization but often lacks synergistic integration with distillation principles. To address this problem, we propose a novel multi-phase knowledge distillation framework with neural architecture search, called MPKD-NAS, which enables collaborative optimization of student model parameters and layer structures during the distillation process. We attempt to extract feature representations from multiple phases of model training by designing an adaptive auxiliary encoder to capture the potential classification distribution of the feature representation. The proposed framework achieves dynamic optimization through a NAS-based mechanism that continuously determines the optimal student layer configuration during distillation. Moreover, we redesign the construction of student models based on neural architecture search strategies and the objective function of the knowledge distillation process based on weighted soft labels. We propose a joint optimization paradigm for model parameters and architecture through the deep coupling of KD and NAS. The key idea of MPKD-NAS is to jointly optimize both the knowledge distillation process and the model architecture search process, achieving an optimal solution for knowledge distillation under limited model parameters. Experiments demonstrate that MPKD-NAS significantly outperforms existing baselines in accuracy-compression trade-offs under equivalent computational constraints, establishing a new methodology for lightweight model design.
>


## Requirements:
- python3.6
- pytorch=='1.9.1'
- numpy
- sklearn
- scipy=='1.5.4'
- mne=='0.23.4'
- tqdm

## Data
We used three public datasets in this study:

- SleepEDF-20 (2013 version)
- [SleepEDF-78](https://physionet.org/content/sleep-edfx/1.0.0/) (2018 version)
- [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)
