# About

*Work in progress*

This repository is the code associated with the engineering thesis "Learning-Based Staining Normalization for Improving Registration of Histology Images".

This presents our try to evaluate deep learning approaches in stain normalization as a preprocessing step in image registration.

# How to use

*Keep in mind we are currently streamlining this process*

For affine alignment configure and run `AffineClicker/main.py`. Click through all the images and save the list.
Then run `AffineClicker/affine_trans.py` to generate a summary table of the dataset with additional affine alignment matrices.

For trainig the deep learning model make your configuration in `HistNet/config.py` and run `HistNet/train.py`

For HPC (SLURM) we provide `run_hpc.slurm` file that lets you configure and easily run training with live tensorboard tunneling through ssh.

# Dataset

Dataset provided by the ANHIR Grand Challenge organizers and is available [here](https://anhir.grand-challenge.org/Data/)

# Dependencies

- PyTorch
- NumPy
- Matplotlib
- Torchio
- Kornia
- Pandas
- OpenCV