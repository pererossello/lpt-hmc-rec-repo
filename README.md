# Observacinones

# To DO:

In start chain, specify the data_ref path
also specify the argdic path, 
or set INVERSE_CRIME to True


- set reconstruction folder name
- set argdic (for forward model specification)
    (either via path or with a function that defaults or whatever)
- set data_reference path




## today:
nan_at_128

problem if seed_int = 62
some initial conditions lead to a nan log likelihood?
Where does it come from?
from the bias or the lpt forward? check that! 

has nothing to do with epsilon
nothing to do with log in nll
just summing n_tr leads equally to nans


nan in LPT!!!
it is not bias bc nan at seed_int=62 apears qequally for powerlaw and hierarchical bias

PROBLEM IN SPHERICAL COLLAPSE!
without muscle without sc corretion



LPT1 works
LPT2 works

## Regarding HMC

Ok, definetely scaling the nll, nlp and kinetic by 1/N**3 makes the sampler much more consistent. (with appropiate reescaing of leapfrog step with N^3)

## Regarding NLL

Bigger L leads to to smaller NLL and grad_nll not because dimensional factors (everything is dimensionaless) but because dm field has less contrast and is more homogeneous, which leads to smaller gradients (thats my intuition but seems true)  

El leapfrog step depende tanto de N como de N_TR.

Empirically it seems that the leapfrog step should be scaled by 1/N**(3/2), as max value of gradients seems to scale with that value.

BUT the std of the grad_nll values remains the same regardless of nll....

# TO DO:

- Implementar una funcion que dado delta o delta_hat, me devuelva los N^3 numeros necesarios para definit la transformada de fourier reducida. 