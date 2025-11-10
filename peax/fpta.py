"""
An implementation of iterative FPTA
"""
import jax
import jax.numpy as jnp

from typing import Callable, List

class LSTQD:
    """
    Run Least squares TD learning
    """
    def __init__(self, basis: List[Callable]) -> None:

        self.basis = basis
        self.m = len(basis)

        key = jax.random.key(0) # Using 0 as the seed

        self.w = jax.random.uniform(key, shape=(self.m**2,))

    def fit(self):
        pass





