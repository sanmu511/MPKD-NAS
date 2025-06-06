# MPKD-NAS
### Multi-Phase Knowledge Distillation Framework based on Neural Architecture Search
## Abstract
> Recently,  deep learning models have achieved remarkable success on sleep stage classification, but these methods rely on a large amount of computing resources, making them difficult to use on devices with limited computing resources. Knowledge distillation can perform model compression by purifying a smaller student model, but it does not consider the impact of network architecture on the performance of the student model. To address this problem, we propose a novel multi-phase knowledge distillation framework with neural architecture search for sleep stage classification, called MPKD-NAS. We attempt to extract feature representations from multiple phases of model training by designing an adaptive auxiliary encoder to capture the potential classification distribution of the feature representation. And we modify the objective function of distillation process based on weighted soft labels. Moreover, we redesign the construction of student models based on neural architecture search strategies. The key idea of MPKD-NAS is to optimize both the knowledge distillation process and the model architecture search process, achieving an optimal solution for model distillation under limited model parameters. Experimental results demonstrate that MPKD-NAS significantly compresses the model size compared to the most advanced baseline methods, without compromising or even enhancing the model's performance in sleep stage classification.
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
