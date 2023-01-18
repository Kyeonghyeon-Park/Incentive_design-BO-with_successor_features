import numpy as np
import torch

from utils import utils_taxi
from utils.utils_all import tile_ravel_multi_index
''' 
Code for getting the lower-level transfer figure. 
num_tests = 1인 tar의 result는 duplicate하여 갯수를 맞춤. 
주의사항으로, ssd에서 저장하는 outcome과 row, column이 반대임.
outcomes_t = {list: 4} [{ndarray: (num_tests, 7500)},  # orr
                        {ndarray: (num_tests, 7500)},  # osc
                        {ndarray: (num_tests, 7500)},  # avg_rew
                        {ndarray: (num_tests, 7500)},  # obj
                        ]
'''
alpha = 0.93

NUM_TESTS = 20

# alpha=0.93 using alpha=1.00.
set_1_paths = [
    "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1238/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1239/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1240/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1241/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1242/7499.tar",
]
# alpha=0.93 using alpha=0.00.
set_2_paths = [
    "../results_taxi_IJCAI/alpha=0.93 using alpha=0.00 (5 seeds)/seed 1239/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=0.00 (5 seeds)/seed 1240/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=0.00 (5 seeds)/seed 1241/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=0.00 (5 seeds)/seed 1242/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 using alpha=0.00 (5 seeds)/seed 1243/7499.tar",
]
# Non-transfer.
set_3_paths = [
    "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1234/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1235/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1236/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1237/7499.tar",
    "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1242/7499.tar",
]

# outcomes_l = [np.empty([0, 7500]),
#               np.empty([0, 7500]),
#               np.empty([0, 7500]),
#               np.empty([0, 7500])]  # [orr, osc, avg_rew, obj]
#
# outcomes_r = [np.empty([0, 7500]),
#               np.empty([0, 7500]),
#               np.empty([0, 7500]),
#               np.empty([0, 7500])]

paths_list = [set_1_paths, set_2_paths, set_3_paths]

outcomes = []

for paths in paths_list:
    outcomes_one_set = [np.empty([0, 7500]),
                        np.empty([0, 7500]),
                        np.empty([0, 7500]),
                        np.empty([0, 7500])]  # [orr, osc, avg_rew, obj]
    # Collect and concatenate outcomes of the results.
    for path in paths:
        dict_one_seed = torch.load(path)
        outcomes_one_seed = dict_one_seed["outcomes_t"]
        for i in range(4):  # outcomes_one_seed is a list which consists [orr, osc, avg_rew, obj].
            outcomes_one_seed[i] = tile_ravel_multi_index(outcomes_one_seed[i], [NUM_TESTS, 7500])
            outcomes_one_set[i] = np.concatenate((outcomes_one_set[i], outcomes_one_seed[i]), axis=0)
    outcomes.append(outcomes_one_set)


# for l in dict_l_list:
#     data_l = torch.load(l)
#     outcomes_l_one_seed = data_l["outcomes_t"]
#     for i in range(4):  # outcomes_l_one_seed is a list which consists [orr, osc, avg_rew, obj].
#         outcomes_l_one_seed[i] = tile_ravel_multi_index(outcomes_l_one_seed[i], [NUM_TESTS, 7500])
#         outcomes_l[i] = np.concatenate((outcomes_l[i], outcomes_l_one_seed[i]), axis=0)
#
# for r in dict_r_list:
#     data_r = torch.load(r)
#     outcomes_r_one_seed = data_r["outcomes_t"]
#     for i in range(4):
#         outcomes_r_one_seed[i] = tile_ravel_multi_index(outcomes_r_one_seed[i], [NUM_TESTS, 7500])
#         outcomes_r[i] = np.concatenate((outcomes_r[i], outcomes_r_one_seed[i]), axis=0)

font_settings = {
    'font_name': 'Times',
    'axis_size': 40,  # 24 for the single graph.
    'legend_size': 40,  # 20 for the single graph.
    'tick_size': 40,  # 20 for the single graph.
}

# utils_taxi.get_plt_final_grayscale_only_obj(outcomes_l, outcomes_r, font_settings=font_settings)
# utils_taxi.get_plt_final_grayscale_only_obj(outcomes[0], outcomes[2], font_settings=font_settings)
utils_taxi.get_plt_final_grayscale_only_obj_three_outcomes(*outcomes, font_settings=font_settings)
