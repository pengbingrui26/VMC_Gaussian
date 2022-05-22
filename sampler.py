import numpy as np
import jax
import jax.numpy as jnp


def Gaussian_sampler(shape, key, mu, sigma):
    x = jax.random.normal(key = key, shape = shape)
    return sigma * x + mu


def Gaussian_prob(z, mu, sigma):
    return 1/(np.sqrt(2*jnp.pi)*sigma) * jnp.exp(-(z-mu)**2/(2*sigma**2))


def masker(x):
    return jnp.array([ xx for xx in x if xx < 1. and xx > 0. ])


def Gaussian_fn(batch_g, mu, key):
    shape = (batch_g, )

    def make_g(sigma):
        g = Gaussian_sampler(shape, key, mu, sigma)
        return g

    def log_prob(sigma):
        g = make_g(sigma) 
        p = Gaussian_prob(g, mu, sigma)
        log_p = jnp.log(p)

        def make_logprob(param):
            logp = jnp.log(Gaussian_prob(g, mu, param))
            return lnp
        grad_logp = jax.jacrev(make_logprob)(sigma)    

        return log_p, grad_logp

    return make_g, log_prob



