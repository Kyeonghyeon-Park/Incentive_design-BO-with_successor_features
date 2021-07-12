# Incentive_design-BO-with_successor_features

Our incentive design framework is bi-level optimization framework.

Framework is (or will be) tested for three environments, which are taxi driver repositioning, cleanup and harvest environment.

## Cleanup environment

This example is based on the open-source implementation of DeepMind's Sequential Social Dilemma (SSD) multi-agent game-theoretic environments [[1]](https://github.com/eugenevinitsky/sequential_social_dilemma_games).

### Environment

To build the incentive designer's problem, we have to set the designer's objective and the designer's decision. In the cleanup environment, we temporally set the designer's objective to the social welfare. In addition, the designer's decisions are inventives to cleaning people and penalties to eating people.

Furthermore, we have to build the feature to use the successor feature concept. 

Based on the above conditions, `map_env.py`, `cleanup.py`, `agent.py`, `env_creator` are modified. 

In addition, `test_envs.py` is modified for testing the modified cleanup environment.

### Lower level

For now (210712), I'm currently working on the lower level process. 

`parsed_args.py` is the setting file for the experiment. In addition to the original setting (`default_args.py`), I added several components for our experiment.

`test_for_ssd.py` is the main file for the lower level experiment. After changing the setting in `parsed_args.py`, you can run this file to find the equilibrium of the lower level. Unfortunately, it doesn't give promising convergence results for now. This file runs the experiment and saves videos, results and data.

`networks_ssd.py` is the networks for the cleanup environment. Unlike the taxi example, it contains actor, critic, and psi simultaneously. My final goal is to build the unified networks file for multiple environments. Boltzmann policy (only critic or psi exists) is not completed yet.

### Upper level

Not done yet

### Utils

`utility_funcs.py` is modified for adding several functions. (draw or save plt, save data, make setting txt, etc.)

## Taxi driver repositioning example

This example is from Shou and Di's paper [[2]](https://arxiv.org/abs/2002.06723).

### Environment

`gird_world.py` defines the Shou and Di's taxi driver repositioning game (grid, state transition, ...)

### Lower level

`main.py` is used for lower level. It uses mean field actor-critic algorithm for finding the equilibrium given alpha. 

`main_psi.py` is used for lower level. It uses mean field actor-psi algorithm for finding the equilibrium given alpha. Specifically, it uses weight (w) and w = \[1,alpha]. 

`actor_critic.py` contains the mean field actor-critic algorithm.

`actor_psi.py` contains the mean field actor-psi algorithm.

You can run `main.py` and `main_psi.py` file to find the equilibrium of specific setting. You should notice that calculating loss in `actor_critic.py` and `actor_psi.py` only works for taxi driver repositioning example. It will be modified.

### Upper level

`bayesian_optimization.py` is used for upper level using alpha and f(alpha)

### Utils

`utils.py` contains several utils (draw plot, print statistics, ...)
