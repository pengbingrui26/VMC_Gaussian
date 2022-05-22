import numpy as np
import jax 
import jax.numpy as jnp
from functools import partial

from free_hamiltonian import H_free
from logpsi import log_psi
from metropolis import random_init, make_E, make_QGT, make_QGT_ED, make_loss
from sampler import Gaussian_fn


def optimize_sigma():
    batch = 1000
    t = 1.
    U = 6.
    Lsite = 2
    N = int(Lsite/2)
    hfree = H_free(Lsite, N, t)
    psi = hfree.get_psi0()

    states_init = random_init(batch, Lsite)

    batch_g = 200
    nthermal = 100
    beta = 2.5

    opt_nstep = 500
    learning_rate = 1e-2 

    mu = 0.3

    make_psi_ratio, make_logpsi, grad_g_logpsi = log_psi(psi, Lsite, N) 
    #states = sample_states(make_psi_ratio, g, states_init, nthermal)
 
    import optax

    optimizer = optax.adam(learning_rate = learning_rate)
    param = jax.random.uniform( key = jax.random.PRNGKey(42), minval = 0.001, maxval = 1. )
    opt_state = optimizer.init(param)

    key = jax.random.PRNGKey(42) 

    def step(param, opt_state, key):
        loss_fn = make_loss(beta, psi, Lsite, N, t, U, nthermal, log_psi, Gaussian_fn, batch_g, mu, key)
 
        grad, loss, E_mean = loss_fn(states_init, param)
        updates, opt_state = optimizer.update(grad, opt_state, param)
        param = optax.apply_updates(param, updates)
        return param, opt_state, grad, loss, E_mean

    for i in range(opt_nstep):
        key_old, key = jax.random.split(key, 2)

        param, opt_state, grad, loss, E_mean = step(param, opt_state, key)
        print('istep, sigma, grad, loss, E_mean:')
        print(i, param, grad, loss, E_mean)
        print('\n') 

    



# run =============================================================

optimize_sigma()


