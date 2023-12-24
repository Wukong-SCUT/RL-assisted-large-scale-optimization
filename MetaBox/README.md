# MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

[![ArXiv](https://img.shields.io/badge/arXiv-2310.08252-b31b1b.svg)](https://arxiv.org/abs/2310.08252)

MetaBox is the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods.


## Installations

You can access all MetaBox files with the command:

```shell
git clone git@github.com:GMC-DRL/MetaBox.git
cd MetaBox
```

## Citing MetaBox

The PDF version of the paper is available [here](https://arxiv.org/abs/2310.08252). If you find our MetaBox useful, please cite it in your publications or projects.

```latex
@inproceedings{metabox,
author={Ma, Zeyuan and Guo, Hongshu and Chen, Jiacheng and Li, Zhenrui and Peng, Guojun and Gong, Yue-Jiao and Ma, Yining and Cao, Zhiguang},
title={MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning},
booktitle = {Advances in Neural Information Processing Systems},
year={2023},
volume = {36}
}
```

## Requirements

`Python` >=3.7.1 with the following packages installed:  

* `numpy`==1.21.2  
* `torch`==1.9.0  
* `matplotlib`==3.4.3  
* `pandas`==1.3.3  
* `scipy`==1.7.1
* `scikit_optimize`==0.9.0  
* `deap`==1.3.3  
* `tqdm`==4.62.3  
* `openpyxl`==3.1.2

## Quick Start

* To obtain the figures in our paper, run the following commands:

  ```shell
  cd for_review
  python paper_experiment.py
  ```

  then corresponding figures will be output to `for_revivew/pics`.

  ---

  The quick usage of the four main running interfaces is listed as follows, in the following command, we specifically take `RLEPSO` as an example.

  Firstly, get into the main code folder, src:

  ```shell
  cd ../src
  ```

* To trigger the entire workflow, including **train, rollout and test**, run the following command:

  ```shell
  python main.py --run_experiment --problem bbob --difficulty easy --train_agent RLEPSO_Agent --train_optimizer RLEPSO_Optimizer
  ```

* To trigger the standalone process of **training**:

  ```shell
  python main.py --train --problem bbob --difficulty easy --train_agent RLEPSO_Agent --train_optimizer RLEPSO_Optimizer 
  ```

* To trigger the standalone process of **testing**:

  ```shell
  python main.py --test --problem bbob --difficulty easy --agent_load_dir agent_model/test/bbob_easy/ --agent_for_cp RLEPSO_Agent --l_optimizer_for_cp RLEPSO_Optimizer --t_optimizer_for_cp DEAP_CMAES Random_search
  ```


## Documentation

For more details about the usage of `MetaBox`, please refer to [MetaBox User's Guide](https://gmc-drl.github.io/MetaBox/).

## Datasets


At present, three benchmark suites are integrated in `MetaBox`:  

* `Synthetic` contains 24 noiseless functions, borrowed from [coco](https://github.com/numbbo/coco):bbob with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Noisy-Synthetic` contains 30 noisy functions, borrowed from [coco](https://github.com/numbbo/coco):bbob-noisy with [original paper](https://www.tandfonline.com/eprint/DQPF7YXFJVMTQBH8NKR8/pdf?target=10.1080/10556788.2020.1808977).
* `Protein-Docking` contains 280 problem instances, which simulate the application of protein docking as a 12-dimensional optimization problem, borrowed from [LOIS](https://github.com/Shen-Lab/LOIS) with [original paper](http://papers.nips.cc/paper/9641-learning-to-optimize-in-swarms).

## Baseline Library

**7 MetaBBO-RL optimizers, 1 MetaBBO-SL optimizer and 11 classic optimizers have been integrated into `MetaBox`.** They are listed below.
<!-- Choose one or more of them to be the baseline(s) to test the performance of your own optimizer. -->

**Supported MetaBBO-RL optimizers**:

|   Name   | Year |                        Related paper                         |
| :------: | :--: | :----------------------------------------------------------: |
| DE-DDQN  | 2019 | [Deep reinforcement learning based parameter control in differential evolution](https://dl.acm.org/doi/10.1145/3321707.3321813) |
|  QLPSO   | 2019 | [A reinforcement learning-based communication topology in particle swarm optimization](https://link.springer.com/article/10.1007/s00521-019-04527-9) |
|  DEDQN   | 2021 | [Differential evolution with mixed mutation strategy based on deep reinforcement learning](https://www.sciencedirect.com/science/article/pii/S1568494621005998) |
|   LDE    | 2021 | [Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient](https://ieeexplore.ieee.org/document/9359652) |
|  RL-PSO  | 2021 | [Employing reinforcement learning to enhance particle swarm optimization methods](https://www.tandfonline.com/doi/full/10.1080/0305215X.2020.1867120) |
|  RLEPSO  | 2022 | [RLEPSO:Reinforcement learning based Ensemble particle swarm optimizer✱](https://dl.acm.org/doi/abs/10.1145/3508546.3508599) |
| RL-HPSDE | 2022 | [Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning](https://www.sciencedirect.com/science/article/pii/S2210650222001602) |

**Supported MetaBBO-SL optimizer**:

|  Name  | Year |                        Related paper                         |
| :----: | :--: | :----------------------------------------------------------: |
| RNN-OI | 2017 | [Learning to learn without gradient descent by gradient descent](https://dl.acm.org/doi/10.5555/3305381.3305459) |

**Supported classic optimizers**:

|         Name          | Year |                        Related paper                         |
| :-------------------: | :--: | :----------------------------------------------------------: |
|          PSO          | 1995 | [Particle swarm optimization](https://ieeexplore.ieee.org/abstract/document/488968) |
|          DE           | 1997 | [Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces](https://dl.acm.org/doi/abs/10.1023/A%3A1008202821328) |
|        CMA-ES         | 2001 | [Completely Derandomized Self-Adaptation in Evolution Strategies](https://ieeexplore.ieee.org/document/6790628) |
| Bayesian Optimization | 2014 | [Bayesian Optimization: Open source constrained global optimization tool for Python](https://github.com/bayesian-optimization/BayesianOptimization) |
|        GL-PSO         | 2015 | [Genetic Learning Particle Swarm Optimization](https://ieeexplore.ieee.org/abstract/document/7271066/) |
|       sDMS_PSO        | 2015 | [A Self-adaptive Dynamic Particle Swarm Optimizer](https://ieeexplore.ieee.org/document/7257290) |
|          j21          | 2021 | [Self-adaptive Differential Evolution Algorithm with Population Size Reduction for Single Objective Bound-Constrained Optimization: Algorithm j21](https://ieeexplore.ieee.org/document/9504782) |
|         MadDE         | 2021 | [Improving Differential Evolution through Bayesian Hyperparameter Optimization](https://ieeexplore.ieee.org/document/9504792) |
|        SAHLPSO        | 2021 | [Self-Adaptive two roles hybrid learning strategies-based particle swarm optimization](https://www.sciencedirect.com/science/article/pii/S0020025521006988) |
|     NL_SHADE_LBC      | 2022 | [NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization](https://ieeexplore.ieee.org/abstract/document/9870295) |
|     Random Search     |  -   |                              -                               |

Note that `Random Search` is to randomly sample candidate solutions from the searching space. 

## Post-processing
In a bid to illustrate the utility of MetaBox for facilitating rigorous evaluation and in-depth analysis, as mentioned in our paper, we carry out a wide-ranging benchmarking study on existing MetaBBO-RL methods. The post-processed data is available in [content.md](post_processed_data/content.md).

<!-- To facilitate the observation of our baselines and related metrics, we tested our baselines on two levels of difficulty on three datasets. Post-processed data are provided in [content.md](post_processed_data/content.md). -->



## Acknowledgements
 
The code and the framework are based on the repos [DEAP](https://github.com/DEAP/deap), [coco](https://github.com/numbbo/coco) and [Protein-protein docking V4.0](https://zlab.umassmed.edu/benchmark/).

