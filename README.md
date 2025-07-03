# TTO

Truss Topology Optimization (TTO) for 2D cellular and monolithic structures, including stress and topological buckling constraints.

⚠️ Note: The current implementation supports 2D structures only.

## Notebooks

Two Jupyter notebooks are provided to run optimizations:

- 2D_launcher.ipynb: for monolithic (non-modular) structures.
- 2D_launcher_modular.ipynb: for modular structures.

These notebooks serve as entry points to the workflow and contain relevant configuration examples.

## Installation

To install the environment using conda, run:

```bash
conda env create -f environment.yml -n TTO
conda activate TTO
```

This will create and activate the TTO environment with all necessary dependencies.

## References

For more details, please refer to the following works:

- PhD Thesis — E. Stragiotti, 2024: <https://hal.science/tel-04824181/>
- Stragiotti, E. et al. (2024). Topology optimization of modular truss-like structures under buckling and stress constraints. Structural and Multidisciplinary Optimization. <https://link.springer.com/article/10.1007/s00158-024-03739-5>
