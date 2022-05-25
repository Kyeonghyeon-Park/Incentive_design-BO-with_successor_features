import numpy as np
import torch

from main_taxi import roll_out
from parsed_args_taxi import args
from utils import utils_all, funcs_taxi

"""
This code is extended version of evaluate_taxi.py for multi-environments and multi-policies. 
You should set alphas_env and paths_pol_dict.
Set alpha which you want to test (i.e., you set w').
paths_pol_dict contains paths of trained policies.
It will save evaluated results. 
"""

# HERE #####
alphas_env = [0, 0.3, 0.43, 0.50, 0.54, 0.62, 0.85, 1]
paths_pol_dict = {
    0.00: "./results/211008 submitted version/results_taxi_final/alpha=0.00/7499.tar",
    # 0.13: "./results_taxi/setting_15/saved/7499.tar",
    0.30: "./results/211008 submitted version/results_taxi_final/alpha=0.30/7499.tar",
    0.43: "./results_taxi/setting_21_220518_1757/saved/7499.tar",
    # 0.45: "./results_taxi/setting_17/saved/7499.tar",
    # 0.47: "./results_taxi/setting_14/saved/7499.tar",
    # 0.53: "./results_taxi/setting_10/saved/7499.tar",
    # 0.50: "./results_taxi/setting_18/saved/7499.tar",
    0.50: "./results_taxi/setting_20_220518_1741/saved/7499.tar",
    # 0.54: "./results_taxi/setting_7/saved/7499.tar",
    0.54: "./results_taxi/setting_22_220519_1158/saved/7499.tar",
    0.62: "./results_taxi/setting_19_220518_1723/saved/7499.tar",
    # 0.73: "./results_taxi/setting_23_220519_1242/saved/7499.tar",
    # 0.64: "./results_taxi/setting_16/saved/7499.tar",
    # 0.81: "./results_taxi/setting_12/saved/7499.tar",
    0.85: "./results_taxi/setting_11/saved/7499.tar",
    1.00: "./results/211008 submitted version/results_taxi_final/alpha=1.00/7499.tar",
}
############

args.setting_name = "setting_evaluation"
paths_pol = list(paths_pol_dict.values())
alphas_pol = list(paths_pol_dict.keys())
num_env = len(alphas_env)
num_pol = len(alphas_pol)
num_tests = 1000

objs = np.zeros([num_env, num_pol])
explanations = [[] for i in range(num_env)]
utils_all.set_random_seed(1236)

for i in range(num_env):
    for j in range(num_pol):
        args.lv_penalty = alphas_env[i]
        path_pol = paths_pol[j]
        dict_pol = torch.load(path_pol)
        args_pol = dict_pol['args']
        env, networks = funcs_taxi.get_env_and_networks(args, dict_pol)

        obj = np.zeros(num_tests)
        print(f"-----------------------------")
        explanation = f"Test alpha={args.lv_penalty:.2f} using previous networks with " \
                      f"alpha={args_pol.lv_penalty:.2f}"
        explanations[i].append(explanation)
        print(explanation)
        print(f"File path: {path_pol}")
        for k in range(num_tests):
            if ((k + 1) * 10) % num_tests == 0:
                print(f"Env.: {i + 1}/{num_env}, Pol.: {j + 1}/{num_pol}, Test: {k + 1}/{num_tests}")
            samples, outcome = roll_out(networks=networks,
                                        env=env,
                                        decayed_eps=0,
                                        is_train=False)
            _, _, _, obj[k] = outcome
        obj_mean = np.mean(obj)
        print(f"Obj mean : {obj_mean:.4f}")
        objs[i][j] = obj_mean

print(objs)

torch.save(
    {
        'x': alphas_env,
        'y': alphas_pol,
        'f': objs,
    },
    'test.tar'
)
