import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)
from apf.models.prgpmf import PRGPMF

import numpy as np
import numpy.random as rn
import scipy.stats as st

def main():
    """Main method."""
    n_genes = 10
    n_cells = 15
    n_feats = 3
    alpha = 0.1
    lambd = 1

    seed = rn.randint(10000)
    rn.seed(seed)
    print('seed: %d' % seed)

    n_threads = 1

    model = PRGPMF(n_genes=n_genes,
                   n_cells=n_cells,
                   n_feats=n_feats,
                   alpha=alpha,
                   lambd=lambd,
                   seed=seed,
                   n_threads=n_threads)

    data_shp = (n_genes, n_cells)
    Y = np.zeros(data_shp, dtype=np.int32)
    
    mask_p = 0.0
    mask = None
    if mask_p > 0:
        mask = rn.binomial(1, mask_p, size=data_shp)
        percent_missing = np.ceil(100 * mask.sum() / float(mask.size))
        print('%d%% missing' % percent_missing)

    data = np.ma.array(Y, mask=mask)
    model._initialize_data(data)

    def get_schedule_func(burnin=0, update_every=1):
        return lambda x: x >= burnin and x % update_every == 0

    schedule = {'Alpha_IK': get_schedule_func(0, 1),
                'A_IK': get_schedule_func(0, 1),
                'Alpha_JK': get_schedule_func(0, 1),
                'P_JK': get_schedule_func(0, 1),
                'mtx_MKD': get_schedule_func(np.inf, 1),
                'core_Q': get_schedule_func(np.inf, 1),
                'Y_MKD': get_schedule_func(0, 1),
                'Y_Q': get_schedule_func(0, 1)}

    var_funcs={}

    model.schein(5000, var_funcs=var_funcs, schedule=schedule)


if __name__ == '__main__':
    main()
