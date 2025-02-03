[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Learning to Defer as a Mixture of Experts

This repository contains a Jax-based implementation of *Learning to defer* using the Expectation - Maximisation (EM) algorithm presented as a baseline in the following paper:

> Cuong Nguyen, Toan Do and Gustavo Carneiro. "**Probabilistic Learning to Defer: Handling Missing Expert Annotations and Controlling Workload Distribution**". In *International Conference on Learning Representations*, volume 13, 2025.

In particular, the *Learning to defer* (or L2D for short) is modelled as a mixture of non-learnable (human) and learnable (classifier) experts. And because mixture of experts is a latent variable, it can be solved efficiently by the EM algorithm.

## Implementation details

### Data specification and loading

The implementation in this repository employs *mlx-data* - a framework-agnostic data loading library from Apple - to load data. Specifically, data is managed through *json* files. Each json file has the following format:

```json
[
    {
        "file": "train/19/bos_taurus_s_000507.png",
        "label": 19,
        "superclass": 11
    },
    {
        "file": "train/29/stegosaurus_s_000125.png",
        "label": 29,
        "superclass": 15
    },
    {
        "file": "train/0/mcintosh_s_000643.png",
        "label": 0,
        "superclass": 4
    },
    {
        "file": "train/11/altar_boy_s_001435.png",
        "label": 11,
        "superclass": 14
    }
]
```

To increase the modularity of the implementation, one json file is used to represent the dataset associated with an expert (either human or classifier).

### Running through *apptainer*

For containerisation purpose, the environment used in the implementation is also provided through the usage of *apptainer*.

#### Why apptainer, not docker?

Because apptainer can run without root access, while docker mostly requires that (although one can setup docker for non-root users). Nevertheless, the usage of apptainer is not too deviated from docker.

#### Build an apptainer image

Please follow the environment setup specified in `apptainer.def` to build a container image.

To build an apptainer image, fire the following command on a Linux terminal:

```
apptainer build apptainer.sif apptainer.def
```

The apptainer will build an image named `apptainer.sif` based on the specifications defined in `apptainer.def`.

#### Install apptainer

Please refer to the [official instruction](https://apptainer.org/docs/admin/main/installation.html) from apptainer.

### Configuration and running

The implementation is configured with `hydra`. Please modify the corresponding files in the `conf` folder to configure the dataset and experiment of interest. For example, to configure an experiment on Cifar-100, one needs to modify the configuration files as follows:
- *conf/conf.yaml*: update the line right after `defaults` to point to the correct `yaml` file representing the dataset of interest, and
- *conf/cifar100.yaml*: the file that is pointed in the main `conf/conf.yaml`. This file contains the information of the dataset and training hyper-parameters.

To run through apptainer, follow the commands and syntax specified in `apptainer.sh`. There are two notes on running with apptainer:
- flag `-nv` to allow apptainer to access to the GPU of the system one wants to run, and
- flag `--bind` to bind the folder containing the dataset of interest, so that apptainer can access to that folder.

### Experiment tracking and management

**MLflow** is used as the experiment tracking and management tool in this implementation. One can track the experiment by 

### Citation

```bibtex
@inproceedings{nguyen2025probabilistic,
    title={Probabilistic Learning to Defer: Handling Missing Expert Annotations and Controlling Workload Distribution},
    author={Nguyen, Cuong and Do, Toan and Carneiro, Gustavo},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=zl0HLZOJC9}
}
```