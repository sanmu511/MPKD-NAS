from caculate_metrics import Caculate_Metrics
from args import Config, Path
config = Config()
path = Path()
caculate_metrics = Caculate_Metrics(config, path, 'student_hkd_sl_Kfold_models')
caculate_metrics._calc_metrics()