from main_taxi import *
from networks_taxi import Networks
from parsed_args_taxi import args
from taxi import TaxiEnv
from utils import utils, utils_taxi

"""
Deprecated. Use evaluate_taxi_multi.py. 
"""

args.setting_name = "setting_evaluation"

##### You should set this part ######
# Set alpha which you want to test (i.e., you set w').
args.lv_penalty = 1.00

# Set network lists which you want to test (i.e., you set previous networks of w_0,...,w_n).
# prev_paths = {"alpha=0.00": "./results_taxi/setting_14/saved/7499.tar",
#               "alpha=0.30": "./results_taxi/setting_13/saved/7499.tar",
#               # "alpha=0.50(X)": "./results_taxi/setting_9/saved/7499.tar",
#               "alpha=0.50": "./results_taxi/setting_17/saved/7499.tar",
#               # "alpha=0.55": "./results_taxi/setting_19/saved/7499.tar",
#               # "alpha=0.56": "./results_taxi/setting_20/saved/7499.tar",
#               # "alpha=0.58": "./results_taxi/setting_21/saved/7499.tar",
#               # "alpha=0.63": "./results_taxi/setting_18/saved/7499.tar",
#               # "alpha=0.70": "./results_taxi/setting_16/saved/7499.tar",
#               "alpha=0.80": "./results_taxi/setting_15/saved/7499.tar",
#               "alpha=1.00": "./results_taxi/setting_10/saved/7499.tar",
#               # "alpha=1.00(X)": "./results_taxi/setting_24/saved/7499.tar",
#               }
# prev_paths = {"alpha=0.00": "./results_taxi/setting_14/saved/7499.tar",
#               "alpha=0.30": "./results_taxi/setting_13/saved/7499.tar",
#               # "alpha=0.50(X)": "./results_taxi/setting_9/saved/7499.tar",
#               "alpha=0.50": "./results_taxi/setting_17/saved/7499.tar",
#               # "alpha=0.55": "./results_taxi/setting_19/saved/7499.tar",
#               # "alpha=0.56": "./results_taxi/setting_20/saved/7499.tar",
#               # "alpha=0.58": "./results_taxi/setting_21/saved/7499.tar",
#               "alpha=0.63": "./results_taxi_final/alpha=0.63 using alpha=0.50/7499.tar",
#               # "alpha=0.70": "./results_taxi/setting_16/saved/7499.tar",
#               "alpha=0.80": "./results_taxi/setting_15/saved/7499.tar",
#               "alpha=1.00": "./results_taxi/setting_10/saved/7499.tar",
#               # "alpha=1.00(X)": "./results_taxi/setting_24/saved/7499.tar",
#               }
# prev_paths = {"alpha=0.80": "./results_taxi_final/alpha=0.63 using alpha=0.50/7499.tar",
#               }
# 220307
prev_paths = {
    "alpha=0.00": "./results_taxi_final/alpha=0.00/7499.tar",
    "alpha=0.30": "./results_taxi_final/alpha=0.30/7499.tar",
    "alpha=0.50": "./results_taxi_final/alpha=0.50/7499.tar",
    "alpha=0.56": "./results_taxi_final/alpha=0.56 using alpha=1.00/7499.tar",
    "alpha=0.63": "./results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1234/7499.tar",
    "alpha=0.80": "./results_taxi_final/alpha=0.80/7499.tar",
    "alpha=1.00": "./results_taxi_final/alpha=1.00/7499.tar",
              }

# Set the number of tests
num_tests = 1000
#####################################

# Seed setting.
utils.set_random_seed(1238)

# Build the environment.
env = TaxiEnv(args)

# Build networks and paths (paths are not used).
networks = Networks(env, args)

# Outcomes.
objs = np.zeros(len(prev_paths))
explanations = []

# Test
prev_paths_list = list(prev_paths.values())
for j in range(len(prev_paths_list)):
    prev_path = prev_paths_list[j]
    dict_trained = torch.load(prev_path)
    prev_args = dict_trained['args']
    # Load
    networks = utils.load_networks(networks, args, dict_trained)

    orr, osc, avg_rew, obj = [np.zeros(num_tests) for _ in range(4)]
    print(f"-----------------------------")
    explanation = f"Test alpha={args.lv_penalty:.2f} using previous networks with " \
                  f"alpha={prev_args.lv_penalty:.2f}"
    explanations.append(explanation)
    print(explanation)
    print(f"File path: {prev_path}")

    for i in range(num_tests):
        print(f"Test num: {i + 1}/{num_tests}") if (((i + 1) * 10) % num_tests == 0) else None
        samples, outcome = roll_out(networks=networks,
                                    env=env,
                                    args=args,
                                    decayed_eps=0,
                                    is_train=False)
        orr[i], osc[i], avg_rew[i], obj[i] = outcome
    print(f"Obj mean : {np.mean(obj):.4f}")
    objs[j] = np.mean(obj)

idx = objs.argmax()
print(f"-----------------------------")
print(f"All results")
for j in range(len(prev_paths_list)):
    print(explanations[j])
    print(f"Obj mean : {objs[j]:.4f}")
print(f"-----------------------------")
print(f"Maximum result")
print(explanations[idx])
print(f"Obj mean : {objs[idx]:.4f}")