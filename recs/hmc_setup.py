import sys
import os
from functools import partial

import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forward_model.energies import nll
from forward_model.energies import nlp_u, kinetic_u, sample_momenta_u, make_is_u
from forward_model.energies import nlp_delta, kinetic_delta, sample_momenta_delta, make_is_delta


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forward_model.main import get_forward_model as get_fm
from forward_model.utils import compute_or_load_pow_spec
from forward_model.utils import compute_or_load_D1_D2
from hmc.metropolis import metropolis_hastings
from hmc.leapfrog import (
    leapfrog_high_order,
    leapfrog_second_order,
    leapfrog_second_order_rev,
)


class HMCSetUp:

    def __init__(self, config, n_tr_ref, argdic, n, l, z_i):

        self.argdic = argdic
        self.n_tr_ref = n_tr_ref
        self.config = config

        self.n = n
        self.l = l
        self.z_i = z_i

        self.inv_l3 = 1 / (self.l**3)
        self.n3 = self.n**3

        self.ic_kind = self.argdic["IC_KIND"]

        self.forward_model = self.get_forward_model()

        if self.ic_kind == "DELTA":
            self.pow_spec = compute_or_load_pow_spec(n, l, z_i)
            self.inv_pow_spec = jnp.where(self.pow_spec != 0.0, 1 / self.pow_spec, 0.0)

        self.get_self_energies()
        self.get_self_hamiltonian()
        self.get_self_grad_potential()
        self.get_self_grad_kinetic()

        return

    def make_initial_states(self, keys):
        if self.ic_kind in ["U", "FSK_U"]:
            make_is = partial(make_is_u, N=self.n)

        elif self.ic_kind == "DELTA":
            make_is = partial(
                make_is_delta,
                N=self.n,
                L=self.l,
                N3=self.n3,
                INV_L3=self.inv_l3,
                pow_spec=self.pow_spec,
            )

        initial_states = []
        for key in keys:
            initial_states.append(make_is(key))

        return initial_states

    def get_metropolis(self):
        return partial(metropolis_hastings, hamiltonian=self.hamiltonian)

    def get_leapfrog(self):
        if self.config.lf_method == "HIGH_ORDER":
            leapfrog = leapfrog_high_order
        elif self.config.lf_method == "STANDARD":
            leapfrog = leapfrog_second_order
        elif self.config.lf_method == "STANDARD_REV":
            leapfrog = leapfrog_second_order_rev

        leapfrog_ = partial(
            leapfrog,
            grad_potential=self.grad_potential,
            grad_kinetic=self.grad_kinetic,
        )

        return leapfrog_

    def get_self_hamiltonian(self):
        potential = lambda q: self.nlp(q) + self.nll(q)
        self.hamiltonian = lambda q, p: potential(q) + self.kinetic(p)

    def get_hamiltonian(self):
        return self.hamiltonian

    def get_self_grad_kinetic(self):
        self.grad_kinetic = jax.grad(self.kinetic)

    def get_self_grad_potential(self):
        potential = lambda q: self.nll(q) + self.nlp(q)
        self.grad_potential = jax.grad(potential)

    def get_grad_nll(self):
        return jax.grad(self.nll)

    def get_grad_nlp(self):
        return jax.grad(self.nlp)

    def get_self_energies(self):

        self.get_nll()
        self.get_nlp()
        self.get_kinetic()
        self.get_sample_momenta()

    def get_nlp(self):
        if self.ic_kind in ["U", "FSK_U"]:
            self.nlp = nlp_u

        elif self.ic_kind == "DELTA":
            self.nlp = partial(
                nlp_delta,
                L=self.l,
                INV_L3=self.inv_l3,
                inv_pow_spec=self.inv_pow_spec,
            )

    def get_kinetic(self):
        if self.ic_kind in ["U", "FSK_U"]:
            self.kinetic = kinetic_u

        elif self.ic_kind == "DELTA":
            self.kinetic = partial(
                kinetic_delta,
                L=self.l,
                INV_L3=self.inv_l3,
                pow_spec=self.pow_spec,
            )

    def get_sample_momenta(self):
        if self.ic_kind in ["U", "FSK_U"]:
            sample_momenta = partial(sample_momenta_u, N=self.n)

        elif self.ic_kind == "DELTA":
            sample_momenta = partial(
                sample_momenta_delta,
                N=self.n,
                L=self.l,
                N3=self.n3,
                INV_L3=self.inv_l3,
                inv_pow_spec=self.inv_pow_spec,
            )
        return sample_momenta

    def get_nll(self, epsilon=1e-10):
        self.nll = partial(
            nll,
            n_tr_ref=self.n_tr_ref,
            forward_model=self.forward_model,
            epsilon=epsilon,
        )

    def get_forward_model(self):
        forward_model = get_fm(self.argdic)
        return forward_model
