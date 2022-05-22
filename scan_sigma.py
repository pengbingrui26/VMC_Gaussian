import numpy as np
import jax 
import jax.numpy as jnp
from functools import partial

from free_hamiltonian import H_free
from logpsi import log_psi
from metropolis import random_init, make_E, make_QGT, make_QGT_ED, make_loss
from sampler import Gaussian_fn


def scan_sigma():
    batch = 500
    t = 1
    U = 6.
    Lsite = 2
    #g = 0.47
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    batch_g = 2000
    nthermal = 100
    beta = 2.5

    mu = 0.3

    make_psi_ratio, make_logpsi, grad_g_logpsi = log_psi(psi, Lsite, N) 
    #states = sample_states(make_psi_ratio, g, states_init, nthermal)
 
    sigmas= np.arange(0.1, 0.31, 0.01)
    loss_all = []
    loss_error_all = []
    grad_all = []
    grad_error_all = []
    E_all = []

    key = jax.random.PRNGKey(42) 

    for param in sigmas: 
        key_old, key = jax.random.split(key, 2)
        loss_fn = make_loss(beta, psi, Lsite, N, t, U, nthermal, \
                               log_psi, Gaussian_fn, batch_g, mu, key)

        grad, grad_error, loss, loss_error, E_mean = loss_fn(states_init, param)
        print('sigma, grad, loss:', param, grad, loss)
        loss_all.append(loss)
        loss_error_all.append(loss_error)
        grad_all.append(grad)
        grad_error_all.append(grad_error)
        E_all.append(E_mean)

    datas = {'U': U, 'batch_g': batch_g, 'beta': beta, 'sigmas': sigmas, \
             'loss': loss_all, 'loss_error': loss_error_all, 'grad': grad_all, 'grad_error': grad_error_all, 'E_mean': E_all}
  
    import pickle as pk
    #fd = open('./scan_sigmas', 'wb')
    #pk.dump(datas, fd)
    #fd.close()
    print('\n')

# run =============================================================

scan_sigma()


