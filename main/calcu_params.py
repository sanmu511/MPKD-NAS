# from model import Transformer
from s_model import Transformer
from args import Config, Path
config = Config()
# model = Transformer(config)
s_model = Transformer(config)
total = sum([param.nelement() for param in s_model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))