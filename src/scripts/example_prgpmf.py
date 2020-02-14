import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)
from apf.models.prgpmf import PRGPMF

import numpy as np
import numpy.random as rn
import scipy.stats as st
import sktensor as skt

import matplotlib.pyplot as plt
import seaborn as sns


def prg(alpha, lambd, size=1):
    """Sample from the Poisson-randomized gamma."""
    shapes = rn.poisson(alpha, size=size)
    gammas = np.zeros_like(shapes)
    gammas[shapes > 0] = rn.gamma(shapes[shapes > 0], 1./lambd)
    return gammas

n_cells = 10  # number of observed cells
n_genes = 30  # number of observed genes
n_feats = 5   # number of latent features

alpha = 1.0   # prior rate of latent poisson counts
lambd = 1.0   # prior rate of latent gamma variables

seed = 617    # random seed (default=None)
n_threads = 4 # number of CPU threads to parallelize over


# Generate a synthetic toy data set
true_A_IK = prg(alpha, lambd, size=(n_genes, n_feats))  # synthetic genes x feats matrix
true_P_JK = prg(alpha, lambd, size=(n_cells, n_feats))  # synthetic cells x feats matrix
true_M_IJ = true_A_IK.dot(true_P_JK.T)                  # synthetic mean of observed counts
true_Y_IJ = np.zeros_like(true_M_IJ, dtype=int)         # synthetic observed counts
true_Y_IJ[true_M_IJ > 0] = rn.poisson(true_M_IJ[true_M_IJ > 0])

subs = true_Y_IJ.nonzero()                    # subscripts where the ndarray has non-zero entries   
vals = true_Y_IJ[true_Y_IJ.nonzero()]         # corresponding values of non-zero entries
sp_data = skt.sptensor(subs,                  # create an sktensor.sptensor 
                       vals,
                       shape=true_Y_IJ.shape,
                       dtype=true_Y_IJ.dtype)

sns.heatmap(true_Y_IJ, cmap='Blues')
plt.show()

model = PRGPMF(n_genes=n_genes,
               n_cells=n_cells,
               n_feats=n_feats,
               alpha=alpha,
               lambd=lambd,
               seed=seed,
               n_threads=n_threads)

n_samples = 100  # how many posterior samples to collect
n_burn_in = 100  # how iterations of burn-in before starting to collect
thin_rate = 10   # how many samples to thin between collected samples

out_dir = Path('samples') # directory to save collected samples to
out_dir.makedirs_p()

verbose = thin_rate       # how often to print information to terminal

# initialize and burn-in the model
model.fit(sp_data, n_itns=n_burn_in, initialize=True, verbose=0)

# run iterations after burn-in
for _ in range(n_samples):
    model.fit(true_Y_IJ, n_itns=thin_rate, initialize=False, verbose=verbose)

    state = dict(model.get_state())                                          # collect sample
    itn_num = model.total_itns                                               # mcmc iteration number
    np.savez_compressed(out_dir.joinpath('state_%d.npz' % itn_num), **state) # serialize sample


# example of how to analyze results
all_samples = out_dir.files('state*.npz')

sample_path = all_samples[0]
itn_num = int(sample_path.namebase.split('_')[1])
print('Inspecting posterior sample from MCMC iteration %d...' % itn_num)

# samples (aka "states") are stored in numpy compressed files
state = np.load(sample_path)  # load them like this
print(state.files)            # see what arrays they contain like this

A_IK = state['A_IK']          # load the inferred genes x features gamma matrix
P_JK = state['P_JK']          # load the inferred cells x features gamma matrix
Alpha_IK = state['Alpha_IK']  # load the inferred genes x features count matrix
Alpha_JK = state['Alpha_JK']  # load the inferred cells x features count matrix

# visually compare the inferred genes x features gamma matrix to the "true" one
# NOTE: due to label-switching, the rows/cols may not be aligned between them
sns.heatmap(A_IK, cmap='Blues')
plt.show()

sns.heatmap(true_A_IK, cmap='Blues')
plt.show()

