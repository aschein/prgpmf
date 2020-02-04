# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
cimport numpy as np

from cython.parallel import parallel, prange

from apf.base.apf cimport APF
from apf.base.sample cimport _sample_gamma, _sample_poisson
from apf.base.sbch cimport _sample as _sample_sbch
from apf.base.cyutils cimport _sum_double_vec, _sum_int_vec, _dot_vec
from apf.base.mcmc_model_parallel import exit_if

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

cdef class PRGPMF(APF):
    """Poisson-Randomized Gamma Poisson Matrix Factorization (PRGPMF)

    Poisson matrix factorization with PRG priors.

    Designed specifically for factorizing gene-cell matrices of RNA sequence counts,
    as in Stein-O’Brien et al. (2019) [1].

    [1] Stein-O’Brien, Genevieve L., et al. "Decomposing cell identity for 
    transfer learning across cellular measurements, platforms, tissues, 
    and species." Cell systems 8.5 (2019): 395-411.
    https://www.cell.com/cell-systems/pdfExtended/S2405-4712(19)30146-2.
    """

    cdef:
        int n_genes, n_cells, n_feats
        double alpha, lambd
        double[:,::1] A_IK, P_JK
        int[:,::1] Alpha_IK, Alpha_JK

    def __init__(self, int n_genes, int n_cells, int n_feats,
                 double alpha, double lambd, object seed=None, object n_threads=None):
        
        super().__init__(data_shp=(n_genes, n_cells),
                         core_shp=(n_feats,),
                         binary=0,
                         mtx_is_dirichlet=[],
                         seed=seed,
                         n_threads=n_threads)
        self.core_Q[:] = 1.

        # Params
        self.n_genes = self.param_list['n_genes'] = n_genes
        self.n_cells = self.param_list['n_cells'] = n_cells
        self.n_feats = self.param_list['n_feats'] = n_feats
        self.alpha = self.param_list['alpha'] = alpha
        self.lambd = self.param_list['lambd'] = lambd

        # State variables
        self.A_IK = np.ones((n_genes, n_feats))
        self.P_JK = np.ones((n_cells, n_feats))

        self.Alpha_IK = np.ones((n_genes, n_feats), dtype=np.int32)
        self.Alpha_JK = np.ones((n_cells, n_feats), dtype=np.int32)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('Alpha_IK', self.Alpha_IK, self._update_Alpha_IK_and_A_IK),
                     ('A_IK', self.A_IK, self._dummy_update),
                     ('Alpha_JK', self.Alpha_JK, self._update_Alpha_JK_and_P_JK),
                     ('P_JK', self.P_JK, self._dummy_update),
                     ('Y_MKD', self.Y_MKD, self._update_Y_PQ),
                     ('Y_Q', self.Y_Q, self._dummy_update)]
        return variables

    def get_default_schedule(self):
        return {}

    def set_state(self, state):
        for key, val, _ in self._get_variables():
            if key in state.keys():
                state_val = state[key]
                assert val.shape == state_val.shape
                for idx in np.ndindex(val.shape):
                    val[idx] = state_val[idx]
        # self._compute_mtx_KT()
        self._update_cache()

    cdef void _initialize_state(self, dict state={}):
        """
        Initialize internal state.
        """
        for key, val, update_func in self._get_variables():
            if key in state.keys():
                state_val = state[key]
                assert val.shape == state_val.shape
                for idx in np.ndindex(val.shape):
                    val[idx] = state_val[idx]
            else:
                output = update_func(self, update_mode=self._INITIALIZE_MODE)
                exit_if(output, 'updating %s' % key)
        # self._compute_mtx_KT()
        self._update_cache()

    cdef int _update_Alpha_IK_and_A_IK(self, int update_mode):
        """
        Jointly sample the "Amplitude" gene-feature matrix and 
        its corresponding latent Poisson shape matrix. 
        """
        cdef:
            np.npy_intp i, k, tid
            double rte_i, rte_ik, r_ik, shp_ik
            double[:,::1] Y_zeta_IK
            long[:,::1] Y_KI
            gsl_rng * rng

        if update_mode in [self._INITIALIZE_MODE, self._GENERATE_MODE]:
            self.A_IK[:] = 0
            self.mtx_MKD[0, :] = 0

            sca = 1. / self.lambd
            for i in range(self.n_genes):
                for k in prange(self.n_feats, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]

                    self.Alpha_IK[i, k] = shp_ik = _sample_poisson(rng, self.alpha)
                    if shp_ik > 0:
                        self.A_IK[i, k] = self.mtx_MKD[0, k, i] = _sample_gamma(rng, shp_ik, sca)
            
            if (np.array(self.Alpha_IK) < 0).any():
                raise ValueError('Poisson rates too large (>2e9).')
        
        elif update_mode == self._INFER_MODE:
            Y_KI = self.Y_MKD[0, :, :self.n_genes]
            Y_zeta_IK = self._compute_zeta_m_DK(0)

            for i in range(self.n_genes):
                rte_i = self.lambd
                for k in prange(self.n_feats, schedule='static', nogil=True):
                    tid = self._get_thread(); rng = self.rngs[tid]

                    r_ik = max(1e-300, self.alpha * (rte_i / (rte_i + Y_zeta_IK[i,k])))
                    if Y_KI[k, i] == 0:
                        self.Alpha_IK[i,k] = _sample_poisson(rng, r_ik)
                    else:
                        self.Alpha_IK[i,k] =  _sample_sbch(rng, Y_KI[k, i], r_ik)

                    if self.Alpha_IK[i,k] < 0:
                        with gil:
                            raise ValueError('alpha_ik < 0')

                    if self.Alpha_IK[i,k] == 0:
                        self.A_IK[i, k] = self.mtx_MKD[0, k, i] = 0
                    else:
                        shp_ik = self.Alpha_IK[i,k] + Y_KI[k, i]
                        rte_ik = rte_i + Y_zeta_IK[i, k]
                        self.A_IK[i, k] = self.mtx_MKD[0, k, i] = _sample_gamma(rng, shp_ik, 1./rte_ik)
        
        for k in range(self.n_feats):
            self.mtx_MK[0, k] = _sum_double_vec(self.mtx_MKD[0, k])

        return 1
        
    cdef int _update_Alpha_JK_and_P_JK(self, int update_mode):
        """
        Jointly sample the "Pattern" cell-feature matrix and 
        its corresponding latent Poisson shape matrix. 
        """
        cdef:
            np.npy_intp j, k, tid
            double rte_j, rte_jk, r_jk, shp_jk
            double[:,::1] Y_zeta_JK
            long[:,::1] Y_KJ
            gsl_rng * rng

        if update_mode in [self._INITIALIZE_MODE, self._GENERATE_MODE]:
            self.P_JK[:] = 0
            self.mtx_MKD[1] = 0
            self.mtx_MK[1] = 0

            sca = 1. / self.lambd
            for j in range(self.n_cells):
                for k in prange(self.n_feats, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]

                    self.Alpha_JK[j, k] = shp_jk = _sample_poisson(rng, self.alpha)
                    if shp_jk > 0:
                        self.P_JK[j, k] = self.mtx_MKD[1,k,j] = _sample_gamma(rng, shp_jk, sca)
            
            if (np.array(self.Alpha_JK) < 0).any():
                raise ValueError('Poisson rates too large (>2e9).')
        
        elif update_mode == self._INFER_MODE:
            Y_KJ = self.Y_MKD[1, :, :self.n_cells]
            Y_zeta_JK = self._compute_zeta_m_DK(1)

            for j in range(self.n_cells):
                rte_j = self.lambd
                for k in prange(self.n_feats, schedule='static', nogil=True):
                    tid = self._get_thread(); rng = self.rngs[tid]

                    r_jk = max(1e-300, self.alpha * (rte_j / (rte_j + Y_zeta_JK[j, k])))

                    if Y_KJ[k, j] == 0:
                        self.Alpha_JK[j,k] =  _sample_poisson(rng, r_jk)
                    else:
                        self.Alpha_JK[j,k] =  _sample_sbch(rng, Y_KJ[k, j], r_jk)

                    if self.Alpha_JK[j,k] < 0:
                        with gil:
                            raise ValueError('alpha_jk < 0')

                    if self.Alpha_JK[j,k] == 0:
                        self.P_JK[j, k] = self.mtx_MKD[1,k,j] = 0
                    else:
                        shp_jk = self.Alpha_JK[j,k] + Y_KJ[k, j]
                        rte_jk = rte_j + Y_zeta_JK[j, k]
                        self.P_JK[j, k] = self.mtx_MKD[1,k,j] = _sample_gamma(rng, shp_jk, 1./rte_jk)
        
        for k in range(self.n_feats):
            self.mtx_MK[1, k] = _sum_double_vec(self.mtx_MKD[1,k])

        return 1
    
    cdef void _update_mtx_MKD(self, int update_mode):
        pass
