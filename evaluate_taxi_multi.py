from main_taxi import *
from parsed_args_taxi import args
from utils import utils, utils_taxi

"""
This code is extended version of evaluate_taxi.py for multi-environments and multi-policies. 
You should set alphas_env and paths_pol_dict.
Set alpha which you want to test (i.e., you set w').
paths_pol_dict contains paths of trained policies.
"""

# HERE #####
alphas_env = [0.00, 0.30, 0.50, 0.56, 0.63, 0.80, 1.00]
# alphas_env = np.linspace(0, 1, 101)
paths_pol_dict = {
    0.00: "./results_taxi_final/alpha=0.00/7499.tar",
    0.30: "./results_taxi_final/alpha=0.30/7499.tar",
    0.50: "./results_taxi_final/alpha=0.50/7499.tar",
    0.56: "./results_taxi_final/alpha=0.56 using alpha=1.00/7499.tar",
    0.63: "./results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1234/7499.tar",
    0.80: "./results_taxi_final/alpha=0.80/7499.tar",
    1.00: "./results_taxi_final/alpha=1.00/7499.tar",
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
utils.set_random_seed(1234)

for i in range(num_env):
    for j in range(num_pol):
        args.lv_penalty = alphas_env[i]
        path_pol = paths_pol[j]
        dict_pol = torch.load(path_pol)
        args_pol = dict_pol['args']
        env, networks = utils_taxi.get_env_and_networks(args, dict_pol)

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

torch.save(
    {
        'x': alphas_env,
        'y': alphas_pol,
        'f': objs,
    },
    'test.tar'
)
