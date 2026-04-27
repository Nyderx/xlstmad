# xLSTMAD: A Powerful xLSTM-based Method for Anomaly Detection \[ICDM 2025\]

This repository contains the improved official implementation of xLSTMAD, the first anomaly detection framework built
upon a full encoder–decoder xLSTM architecture for multivariate time series (see [paper](https://arxiv.org/abs/2506.22837)).
xLSTMAD was published at the ICDM 2025 (IEEE International Conference on Data Mining).

List of changes in comparison to the original version:

- We improved the implementation to avoid bottlenecks, making xLSTMAD much faster.
- We leverage PyTorch Lightning to make it as easy to use as possible.
- For simplicity, this version focuses only on the reconstruction variant and the MSE error.

The original implementation, along with the experimental setup for TSB-AD-M is available [here](https://github.com/Nyderx/xlstmad/tree/v1-submission-version).

The paper is available here: [https://arxiv.org/pdf/2506.22837](https://arxiv.org/pdf/2506.22837).
When using, please cite:
```
   @INPROCEEDINGS{xlstmad,
    author={Faber, Kamil and Pietron, Marcin and Zurek, Dominik and Corizzo, Roberto},
    booktitle={2025 IEEE International Conference on Data Mining (ICDM)},
    title={xLSTMAD: A Powerful xLSTM-based Method for Anomaly Detection},
    year={2025},
    volume={},
    number={},
    pages={247-256},
    doi={10.1109/ICDM65498.2025.00032}}
```

## Installation

1. Create python environment or use an existing one (we tested xLSTMAD with Python 3.11)
1. Download pytorch and torchvision from [here](https://pytorch.org/get-started/locally/) and install them according to
   your system configuration.
1. Install required dependencies by running:

  ```bash
  pip install -r requirements.txt
  ```

## Example usage

The main code of xLSTMAD method is in `xlstmad.py` file.
Our example script trains xLSTMAD on the example dataset and evaluates it on the test set. Its main goal is to showcase
how to use the `xLSTMAD` class (pytorch lightning module) and how to prepare the data for it.

You can run it by executing the following command:
`python example.py`

The script accepts just one parameter `--slstmbackend` that determines which backend to use for the SLSTM cell. 
The available options are `cuda` and `vanilla`. The `cuda` backend is much faster but requires NVIDIA Compute Capability ≥ 8.0.
The `vanilla` backend is slower but can run on any system without additional dependencies.
