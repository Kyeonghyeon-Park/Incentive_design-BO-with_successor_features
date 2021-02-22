# Incentive_design-BO-with_successor_features

main.py is used for lower level (finding the equilibrium given alpha) 

bayesian_optimization.py is used for upper level using alpha and f(alpha)

gird_world.py defines the Shou and Di's taxi driver repositioning game (grid, state transition, ...)

actor_critic.py contains the mean field actor-critic algorithm

utils.py contains several utils (draw plot, print statistics, ...)


## Update log summary
210217 

Extract some utils of `actor_critic.py` to clean the code (`utils.py`)

Add various explanations in `grid_world.py`, `actor_critic.py`, `bayesian_optimization.py`, `utils.py` and `main.py`

Add some to-do lists in `bayesian_optimization.py` and `actor_critic.py`

Modify the step function to return the figure in `grid_world.py`
