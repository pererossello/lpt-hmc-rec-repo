import jax
import jax.numpy as jnp


def leapfrog_high_order(
    q,
    p,
    LF_STEP_MIN,
    LF_STEP_MAX,
    LF_STEPS, 
    grad_potential,
    grad_kinetic,
    sk_LF_STEP,
):

    # randomize leapfrog step size
    #LF_STEP = LF_STEP_MIN + jax.random.uniform(sk_LF_STEP) * (LF_STEP_MAX - LF_STEP_MIN)
    log_min = jax.numpy.log10(LF_STEP_MIN)
    log_max = jax.numpy.log10(LF_STEP_MAX)
    random_log = jax.random.uniform(sk_LF_STEP, minval=log_min, maxval=log_max)
    LF_STEP = 10**random_log

    def body_fun(i, state):
        q_new, p_new, fact = state

        LF_STEP_PRIME = LF_STEP * fact
        p_half_step = p_new - 0.5 * LF_STEP_PRIME * grad_potential(q_new)
        q_new = q_new + LF_STEP_PRIME * grad_kinetic(p_half_step)
        p_new = p_half_step - 0.5 * LF_STEP_PRIME * grad_potential(q_new)
        return q_new, p_new, fact

    n = 2  # order, n_steps

    fact = 1.
    q_new, p_new, _ = jax.lax.fori_loop(0, LF_STEPS, body_fun, (q, p, fact))

    fact = -(2*LF_STEPS)**(1/(n+1))
    q_new, p_new, _ = jax.lax.fori_loop(0, 1, body_fun, (q_new, p_new, fact))

    fact = 1.
    q_new, p_new, _ = jax.lax.fori_loop(0, LF_STEPS, body_fun, (q_new, p_new, fact))

    return q_new, p_new, LF_STEP



def leapfrog_second_order(
    q,
    p,
    LF_STEP_MIN,
    LF_STEP_MAX,
    LF_STEPS, 
    grad_potential,
    grad_kinetic,
    sk_LF_STEP,
):

    log_min = jax.numpy.log10(LF_STEP_MIN)
    log_max = jax.numpy.log10(LF_STEP_MAX)
    random_log = jax.random.uniform(sk_LF_STEP, minval=log_min, maxval=log_max)
    LF_STEP = 10**random_log

    def body_fun(i, state):
        q_new, p_new = state
        p_half_step = p_new - 0.5 * LF_STEP * grad_potential(q_new)
        q_new = q_new + LF_STEP * grad_kinetic(p_half_step)
        p_new = p_half_step - 0.5 * LF_STEP * grad_potential(q_new)
        return q_new, p_new

    q_new, p_new = jax.lax.fori_loop(0, LF_STEPS, body_fun, (q, p))
    return q_new, p_new, LF_STEP



def leapfrog_second_order_rev(
    q,
    p,
    LF_STEP_MIN,
    LF_STEP_MAX,
    LF_STEPS, 
    grad_potential,
    grad_kinetic,
    sk_LF_STEP,
):

    log_min = jax.numpy.log10(LF_STEP_MIN)
    log_max = jax.numpy.log10(LF_STEP_MAX)
    random_log = jax.random.uniform(sk_LF_STEP, minval=log_min, maxval=log_max)
    LF_STEP = 10**random_log

    def body_fun(i, state):
        q_new, p_new = state
        q_mid = q_new + 0.5 * LF_STEP * grad_kinetic(p_new)
        p_new = p_new - LF_STEP * grad_potential(q_mid)
        q_new = q_mid + 0.5 * LF_STEP * grad_kinetic(p_new)
        return q_new, p_new

    q_new, p_new = jax.lax.fori_loop(0, LF_STEPS, body_fun, (q, p))
    return q_new, p_new, LF_STEP