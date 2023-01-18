import numpy as np
import torch

from utils import utils_ssd
from utils.utils_all import tile_ravel_multi_index
''' 
Code for getting the lower-level transfer figure. 
'''
# alpha value.
alpha = 0.05

NUM_TESTS = 100

# alpha=0.05 using alpha=0.00.
set_1_paths = [
    "../results_ssd_IJCAI/alpha=0.05 using alpha=0.00 or alpha=0.08 (3 seeds)/seed 1234 (using alpha=0.00)/evaluation_results_ssd.tar",
    "../results_ssd_IJCAI/alpha=0.05 using alpha=0.00 or alpha=0.08 (3 seeds)/seed 1267 (using alpha=0.08)/evaluation_results_ssd.tar",
    "../results_ssd_IJCAI/alpha=0.05 using alpha=0.00 or alpha=0.08 (3 seeds)/seed 1268 (using alpha=0.00)/evaluation_results_ssd.tar",
]

# alpha=0.05 using alpha=1.00.
set_2_paths = [
    "../results_ssd_IJCAI/alpha=0.05 using alpha=1.00 (3 seeds)/seed 1234/evaluation_results_ssd.tar",
    "../results_ssd_IJCAI/alpha=0.05 using alpha=1.00 (3 seeds)/seed 1235/evaluation_results_ssd.tar",
]

# Non-transfer.
set_3_paths = [
    "../results_ssd_IJCAI/alpha=0.05 (3 (of 6) seeds)/seed 1235/evaluation_results_ssd.tar",
    "../results_ssd_IJCAI/alpha=0.05 (3 (of 6) seeds)/seed 1239/evaluation_results_ssd.tar",
]


# 31 or 30.
num_networks = 31
# outcomes_1 = np.empty([0, num_networks])
# outcomes_2 = np.empty([0, num_networks])
# outcomes_3 = np.empty([0, num_networks])

paths_list = [set_1_paths, set_2_paths, set_3_paths]

outcomes = []

for paths in paths_list:
    outcomes_one_set = np.empty([0, num_networks])
    # Collect and concatenate outcomes of the results.
    for path in paths:
        dict_one_seed = torch.load(path)
        outcomes_one_seed = dict_one_seed[alpha][alpha]["obj"]
        outcomes_one_seed = tile_ravel_multi_index(outcomes_one_seed, [NUM_TESTS, num_networks])
        outcomes_one_set = np.concatenate((outcomes_one_set, outcomes_one_seed), axis=0)
    outcomes.append(outcomes_one_set)

outcomes_transpose = [np.transpose(outcomes_one_set) for outcomes_one_set in outcomes]

# for first in dict_1_list:
#     dict_1 = torch.load(first)
#     outcomes_1_one_seed = dict_1[alpha][alpha]["obj"]
#     outcomes_1_one_seed = tile_ravel_multi_index(outcomes_1_one_seed, [NUM_TESTS, num_networks])
#     outcomes_1 = np.concatenate((outcomes_1, outcomes_1_one_seed), axis=0)

# Collect and concatenate outcomes of the non-transfer results.
# for r in dict_r_list:
#     dict_r = torch.load(r)
#     outcomes_r_one_seed = dict_r[alpha][alpha]["obj"]
#     outcomes_r_one_seed = tile_ravel_multi_index(outcomes_r_one_seed, [NUM_TESTS, num_networks])
#     outcomes_r = np.concatenate((outcomes_r, outcomes_r_one_seed), axis=0)

# outcomes_l_transpose = np.transpose(outcomes_l)
# outcomes_r_transpose = np.transpose(outcomes_r)

font_settings = {
    'font_name': 'Times',
    'axis_size': 40,  # 24 for the single graph.
    'legend_size': 40,  # 20 for the single graph.
    'tick_size': 40,  # 20 for the single graph.
}

utils_ssd.get_plt_final_aggregate_grayscale_three_outcomes(*outcomes_transpose, font_settings=font_settings)

# # Print average outcomes
# print(f'Transfer outcome    : {np.mean(outcomes_l_transpose[-1, :]):.2f}')
# print(f'Non-transfer outcome: {np.mean(outcomes_r_transpose[-1, :]):.2f}')
