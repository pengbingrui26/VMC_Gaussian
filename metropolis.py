import numpy as np
import jax 
import jax.numpy as jnp
from functools import partial

from ED_dimer import dimer

def count_double_occ(state):
    Lsite = state.shape[-1]
    N = int(Lsite/2)
    up = state[None, :N]
    down = state[N:, None]
    diff = up + Lsite - down
    tmp = jnp.where(diff == 0, 1, 0)
    return jnp.sum(tmp)


def all_hop(state):
    Lsite = state.shape[-1]
    N = int(Lsite/2)
    up, down = state[:N], state[N:] - Lsite
 
    up_copy = jnp.tile(up, (N*2, 1))
    down_copy = jnp.tile(down, (N*2, 1))
     
    hop_matr = jnp.vstack( (jnp.eye(N), -jnp.eye(N) ) ).astype('int32')
    up_hopped = (up_copy + hop_matr) % Lsite
    down_hopped = (down_copy + hop_matr) % Lsite

    hopped_up = jnp.hstack( (up_hopped, down_copy+Lsite) )
    hopped_down = jnp.hstack( (up_copy, down_hopped+Lsite))
    hopped = jnp.vstack( (hopped_up, hopped_down) )
    return hopped


def jump(state, key):
    """
    change a spin state x to a new spin state x'. 
    state has shape (Lsite/2, )
    """
    Lsite = state.shape[-1] * 2
    tmp = state[None,:] - jnp.arange(Lsite)[:,None]
    pro = abs( jnp.prod(tmp, -1) )
    indice = jnp.argsort(pro)
    indice = indice[int(Lsite/2):]

    key_x, key_proposal = jax.random.split(key, 2)
    x = jax.random.choice(key = key_x, a = state)
    x_proposal = jax.random.choice(key = key_proposal, a = indice)
    state_new = jnp.where(state==x, x_proposal, state)

    return state_new 


#@partial(jax.vmap, in_axes = (0, None), out_axes = 0)
def random_init(batch, Lsite):
    """
    randomly initialize a batch of states for the begining of Metropolis
    """
    N = int(Lsite/2)
    import random
    states = []
    for ibatch in range(batch):
        up = random.sample(range(Lsite), N)
        down = random.sample(range(Lsite, Lsite*2), N)
        state = up + down
        states.append(state)
    return jnp.array(states)


def random_init_multi(W, batch, Lsite):
    states_all = []
    for i in range(W):
        states_all.append(random_init(batch, Lsite))
    return jnp.array(states_all)


#@partial(jax.vmap, in_axes =(None, None, 0, 0), out_axes = 0)
def walk(states, make_psi_ratio, params, key_others):
    batch, Lsite = states.shape[-2], states.shape[-1]
    key_up, key_down, key_accept, key_bernoulli = jax.random.split(key_others, 4)

    states_up = states[:, :int(Lsite/2)]
    states_down = states[:, int(Lsite/2):] - Lsite

    key_ups = jax.random.split(key_up, batch)
    key_downs = jax.random.split(key_down, batch)

    jump_vmap = jax.vmap(jump, in_axes = (0, 0), out_axes = 0)

    states_up_proposal = jump_vmap(states_up, key_ups)
    states_down_proposal = jump_vmap(states_down, key_downs) + Lsite

    up_proposal = jnp.hstack((states_up_proposal, states_down + Lsite))
    down_proposal = jnp.hstack((states_up, states_down_proposal))

    keys_bernoulli = jax.random.split(key_bernoulli, batch)
    up_or_down = jax.random.bernoulli(key_bernoulli, p=0.5, shape = (batch,))

    states_proposal = jnp.where(up_or_down[:, None], up_proposal, down_proposal)

    ratio = jax.vmap(make_psi_ratio, in_axes = (0, 0, None), out_axes = 0)(states, states_proposal, params)
    ratio = ratio.real**2 + ratio.imag**2

    accept = jax.random.uniform(key = key_accept, shape = ratio.shape) < ratio

    states = jnp.where(accept[:, None], states_proposal, states)
    return states


def sample_states(make_psi_ratio, params, states_init, nthermal):
    states = jnp.array(list(states_init))
    key = jax.random.PRNGKey(42)

    for istep in range(nthermal):
        #print('istep:', istep)
        key, key_others = jax.random.split(key, 2)
        states = walk(states, make_psi_ratio, params, key_others)
    return states


def make_eloc(t, U, state, make_psi_ratio, params):
    """
    Evaluated as \sum_{x'} < x |H| x' > <x'|Psi>/<x|Psi>, 
    """ 
    all_hopped = all_hop(state)
    ratio = jax.vmap(make_psi_ratio, in_axes = (None, 0, None), out_axes = 0)(state, all_hopped, params)
 
    kinetic = -t * ratio
    kinetic = kinetic.sum(-1)
    if state.shape[-1] == 2:
        kinetic = kinetic/2.
 
    potential = U * count_double_occ(state)
    eloc = kinetic + potential
    return eloc.real


def make_E(t, U, states_init, make_psi_ratio, make_logpsi, g, nthermal):
    states = sample_states(make_psi_ratio, g, states_init, nthermal)

    E_all = jax.vmap(make_eloc, in_axes = (None, None, 0, None, None), out_axes = 0)(t, U, states, make_psi_ratio, g)
    E_mean = E_all.mean()
    
    def grad_E(z):
        ln_p = jax.vmap(make_logpsi, in_axes = (0, None), out_axes = 0)(states, z)
        E_fn = jnp.multiply(ln_p, E_all-E_mean).mean()
        return E_fn
    
    grad = jax.grad(grad_E)(g)

    return E_mean, grad



def make_QGT(t, U, states_init, make_psi_ratio, make_grad_logpsi, g, nthermal):
    states = sample_states(make_psi_ratio, g, states_init, nthermal)

    make_grad_logpsi_vmapped = jax.vmap(make_grad_logpsi, in_axes = (0, None), out_axes = 0)
    grad_logpsi = make_grad_logpsi_vmapped(states, g)
    qgt = grad_logpsi.std()**2
    return qgt 


def make_QGT_ED(t, U, g): 
    model = dimer(t, U)
    qgt = model.qgt(g)
    return qgt.real


def make_Veff(beta, t, U, states_init, make_psi_ratio, make_grad_logpsi, g, nthermal):
    states = sample_states(make_psi_ratio, g, states_init, nthermal)

    E_all = jax.vmap(make_eloc, in_axes = (None, None, 0, None, None), out_axes = 0)(t, U, states, make_psi_ratio, g)
    E_mean = E_all.mean()
    #print('E_mean:')
    #print(E_mean)

    #make_grad_logpsi_vmapped = jax.vmap(make_grad_logpsi, in_axes = (0, None), out_axes = 0)
    #grad_logpsi = make_grad_logpsi_vmapped(states, g)
    #qgt = grad_logpsi.std()**2
 
    #qgt = jax.vmap(make_QGT_ED, in_axes = (None, None, 0), out_axes = 0)(t, U, g)
    qgt = make_QGT_ED(t, U, g)    
    #print('qgt:')
    #print(qgt)

    epsilon = 1e-6
    #qgt += epsilon
    Veff = E_mean - 1/(2*beta) * jnp.log(qgt)
    return Veff, E_mean


def make_loss(beta, psi, Lsite, N, t, U, nthermal, log_psi, Gaussian_fn, batch_g, mu, key):

    make_psi_ratio, make_logpsi, make_grad_logpsi = log_psi(psi, Lsite, N)

    make_g, log_prob = Gaussian_fn(batch_g, mu, key)

    def loss_fn(states_init, sigma):
        gs = make_g(sigma)
        #print('gs_mean, gs_std:', gs.mean(), gs.std())

        Veff, E = jax.vmap(make_Veff, in_axes = (None, None, None, None, None, None, 0, None), \
                     out_axes = 0)(beta, t, U, states_init, make_psi_ratio, make_grad_logpsi, gs, nthermal)

        log_p, grad_logp = log_prob(sigma)

        F = 1/beta * log_p + Veff
        F_mean = F.mean()
        F_std = F.std()
        F_error = F_std/jnp.sqrt(F_std.size)
        E_mean = E.mean()

        assert grad_logp.shape == F.shape
        grads = jnp.multiply(grad_logp, F - F_mean)
        grads = jnp.multiply(grad_logp, F)

        grad = grads.mean()

        return grad, F_mean, E_mean

    return loss_fn

