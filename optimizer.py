import easydict
from Answer import SoyuPredictor
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import numpy as np
import torch

space = [
    Real(10**-5, 10**-3, name='learning_rate'),
    Integer(2, 4, name='num_epochs')
]

@use_named_args(space)
def objective(**params):
    args = easydict.EasyDict(params)
    args.batch_size=100
    args.num_workers=1
    torch.manual_seed(1021)
    torch.cuda.manual_seed(1021)
    np.random.seed(1021)

    predictor = SoyuPredictor(args)
    predictor.run()
    pridictor.test()
    return 1.0 - predictor.accuracy()

res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)
print(res_gp)
print("Best: %.4f" % (1.0 - res_gp.fun))
print(res_gp.x)


