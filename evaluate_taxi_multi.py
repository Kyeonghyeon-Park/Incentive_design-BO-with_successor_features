import time

import numpy as np
import torch

from main_taxi import roll_out
from parsed_args_taxi import args
from utils import utils_all, funcs_taxi

"""
This file is for getting evaluation results given alphas and policies. 
(This code is extended version of evaluate_taxi.py for multi-environments and multi-policies.)
It requires "alphas_env", list that contains alphas of environments, 
and "paths_pol_dict", dict that contains the directory of the file. 
It run "num_tests" times to get results for each alpha of environment and alpha of policy.
Set alphas_env and paths_pol_dict in line 25.

ex. 
alphas_env = [0.00, 0.50, 1.00]

paths_pol_dict = {
    0.00: "./folder_name/7499.tar",
    0.50: "./folder_name/7499.tar",
    1.00: "./folder_name/7499.tar",
}
"""
'''
alpha = 0.43
'''
# alpha_start = 0.43
# alphas_env = [0.00, 0.26, 0.43, 0.50, 0.52, 0.57, 0.70, 1.00]
'''
alpha = 0.63
'''
# alpha_start = 0.63
# alphas_env = [0.00, 0.21, 0.35, 0.45, 0.49, 0.54, 0.55, 0.56, 0.57, 0.63, 0.81, 1.00]
'''
alpha = 0.93
'''
alpha_start = 0.93
# alphas_env = [0.00, 0.16, 0.28, 0.36, 0.41, 0.49, 0.54, 0.57, 0.62, 0.68, 0.73, 0.79, 0.93, 1.00]
alphas_env = [0.00, 0.16, 0.26, 0.41, 0.49, 0.51, 0.54, 0.58, 0.62, 0.79, 0.93]
'''
Single alpha
'''
# alpha_start = 0.41
# alphas_env = [alpha_start]

paths_pol_dict = {alpha: "./results_taxi_IJCAI/lists of policies/"+"{:.2f}".format(alpha)+"/7499.tar" for alpha in alphas_env}
filename = '{:.2f}'.format(alpha_start)+'_'+str(len(alphas_env)-1)+'_'+','.join('{:.2f}'.format(alpha) for alpha in alphas_env)+'.tar'

args.setting_name = "setting_evaluation"
paths_pol = list(paths_pol_dict.values())
alphas_pol = list(paths_pol_dict.keys())
num_env = len(alphas_env)
num_pol = len(alphas_pol)
num_tests = 1000

objs = np.zeros([num_env, num_pol])
explanations = [[] for i in range(num_env)]
utils_all.set_random_seed(1236)

time_start = time.time()
print(f"Evaluation start time: {time.strftime('%y%m%d_%H%M_%S', time.localtime(time_start))}")

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
                print(f"Env.: {alphas_env[i]} ({i + 1}/{num_env}), "
                      f"Pol.: {alphas_pol[j]} ({j + 1}/{num_pol}), "
                      f"Test: ({k + 1}/{num_tests})")
            samples, outcome = roll_out(networks=networks,
                                        env=env,
                                        decayed_eps=0,
                                        is_train=False)
            _, _, _, obj[k] = outcome
        obj_mean = np.mean(obj)
        print(f"Obj mean : {obj_mean:.4f}")
        objs[i][j] = obj_mean

print(f"-----------------------------")
print(f"Start time    : "+time.strftime('%y%m%d_%H%M_%S', time.localtime(time_start)))
print(f"Finished time : "+time.strftime('%y%m%d_%H%M_%S', time.localtime(time.time())))
print(f"Total time spent : {time.time() - time_start:.2f}s")

print(f"List of alphas: {alphas_env}")
print(f"Objective values: ")
print(objs)

start_time_tag = "_"+time.strftime('%y%m%d_%H%M_%S', time.localtime(time_start))

torch.save(
    {
        'x': alphas_env,
        'y': alphas_pol,
        'f': objs,
    },
    filename
)
