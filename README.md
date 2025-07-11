# Duqling Emulators

This package investigates Bayesian Optimization (BO) for Neural Architecture Search (NAS) in designing emulators for uncertainty quantification (UQ) tasks.

The surrogate models are trained on benchmark functions from the [Duqling](https://github.com/knrumsey/duqling) UQ test library by K. Rumsey et al.

### Relevant Papers:
- (2018) - [Neural Architecture Search with Bayesian Optimisation and Optimal Transport](https://arxiv.org/abs/1802.07191)
- (2021) - [BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search](https://arxiv.org/pdf/1910.11858)


### Todo:

- build a more nuanced NAS
    - [ ] support intra-architecture variability of layer shapes
    - [ ] allow different activations across layers
- [ ] refactor and organize file structure

#### Optional:
- [ ] fine tune GPs
- [ ] implement PCE models
- [ ] reimplement decision tree models
