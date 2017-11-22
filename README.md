## Weak Lensing 

# Code developed by Grigor Aslayan and Benjamin Horowitz

Utilities:

1) lin_lbfgs_2d: Generates fake shear-shear fields (or just density) given a mask, anisotropic noise, and input powerspectra. It then reconstructs the initial density field using LBFGS to maximize the likelihood.

2) gamma_gamma: Inputs observed shear-shear fields, mask, and noise and reconstructs the reconstructs the initial density field using LBFGS to maximize the likelihood. (as of 11/21/17: not yet fully implemented)