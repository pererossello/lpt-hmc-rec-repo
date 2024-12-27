import jax


def hmc_sample(
    q,
    key,
    sample_momenta,
    leapfrog,
    metropolis,
):
    # Split key for momentum and Metropolis step
    sk_p, sk_m, sk_LF_STEP = jax.random.split(key, 3)

    # Sample initial momentum
    p = sample_momenta(sk_p)

    # Perform leapfrog integration
    q_new, p_new, lf_step = leapfrog(q=q, p=p, sk_LF_STEP=sk_LF_STEP)

    # Perform Metropolis-Hastings step
    q_next, H_next, accept, alpha, Delta_H = metropolis(q, q_new, p, p_new, sk_m)

    return q_next, accept, alpha, H_next, Delta_H, lf_step


def hmc_sample_epochs(
    q,
    key,
    epoch,
    sample_momenta,
    leapfrog,
    metropolis,
    lf_steps_min,
    lf_steps_max,
    lf_steps,
    temperatures,
):

    # Split key for momentum and Metropolis step
    sk_p, sk_m, sk_LF_STEP = jax.random.split(key, 3)

    # Sample initial momentum
    p = sample_momenta(sk_p)

    # Perform leapfrog integration
    q_new, p_new, lf_step = leapfrog(
        q=q,
        p=p,
        sk_LF_STEP=sk_LF_STEP,
        LF_STEP_MIN=lf_steps_min[epoch],
        LF_STEP_MAX=lf_steps_max[epoch],
        LF_STEPS=lf_steps[epoch],
    )

    # Perform Metropolis-Hastings step
    q_next, H_next, accept, alpha, Delta_H = metropolis(
        q, q_new, p, p_new, sk_m, temperature=temperatures[epoch]
    )

    return q_next, accept, alpha, H_next, Delta_H, lf_step
