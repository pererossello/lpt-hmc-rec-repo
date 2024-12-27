import jax
import jax.numpy as jnp

def metropolis_hastings(q, q_new, p, p_new, key, hamiltonian, temperature):
    H_i = hamiltonian(q, p)
    H_f = hamiltonian(q_new, p_new)
    
    # Compute the change in Hamiltonian
    delta_H_over_T = (H_f - H_i) / temperature
    # Compute the acceptance probability
    alpha = jnp.exp(-delta_H_over_T)
    
    acceptance_prob = jnp.minimum(1.0, alpha)

    # Generate a random uniform number
    u = jax.random.uniform(key)
    # Determine whether to accept the proposal
    accept = u < acceptance_prob

    # Select the next state based on acceptance
    q_next = jax.lax.select(accept, q_new, q)
    H_next = jax.lax.select(accept, H_f, H_i)

    return q_next, H_next, accept, alpha, delta_H_over_T