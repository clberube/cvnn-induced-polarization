# Complex-values neural networks for spectral induced polarization applications

## Necessary libraries (there may be more)
- scikit-learn https://scikit-learn.org/stable/index.html 
- pytorch https://pytorch.org/ 
- matplotlib https://matplotlib.org/ 
- numpy https://numpy.org/ 
- tqdm https://tqdm.github.io/ 

## Code structure
Each experiment is contained within its own directory:
- Experiment I: classip (classification of induced polarization spectra)
- Experiment II: parestim  (Cole-Cole parameter estimation)
- Experiment III: fapprox (mechanistic function approximation)

## How to run
In each experiment directory, the scripts should be run in order
0. Run `##_gen_data.py`
0. Run `train_mlp.py` 
0. Run the other scripts in order

## Other modules in each experiment directory
- `utilities.py` utility functions
- `plotlib.py` plotting helpers
- `models.py` neural networks 
- `seg.mplstyle` my matplotlib style sheet 
