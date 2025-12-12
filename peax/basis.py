import jax.numpy as jnp


def simple_pursuer_evader_basis(alpha, num_trait, num_actions):
    """
    e^x - These set of basis just look at 
    """
    basis = [lambda x: jnp.array(1)]
    basis += [lambda x, idx=i: jnp.exp(alpha*(1 - 2*x[0])*x[idx]) for i in range(num_trait)]
    basis += [lambda x, idx=i: jnp.exp(alpha*(1 - 2*x[0])*x[idx]) for i in range(num_actions)]
    return basis