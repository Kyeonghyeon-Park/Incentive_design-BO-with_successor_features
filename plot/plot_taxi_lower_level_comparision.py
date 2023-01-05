import numpy as np
import torch

from utils import utils_taxi


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
def tile_rav_mult_idx(a, dims):
    """
    https://stackoverflow.com/questions/26374634/numpy-tile-a-non-integer-number-of-times

    Examples
    --------
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = tile_rav_mult_idx(a, [3, 4])
    c = tile_rav_mult_idx(a, [1, 4])
    ->
    b = np.array([[1, 2, 3, 1],
                  [4, 5, 6, 4],
                  [1, 2, 3, 1]])
    c = np.array([[1, 2, 3, 1]])

    Parameters
    ----------
    a: numpy.ndarray
    dims: list

    Returns
    -------
    a_tiled: numpy.ndarray
    """
    a_tiled = a.flat[np.ravel_multi_index(np.indices(dims), a.shape, mode='wrap')]

    return a_tiled


NUM_TESTS = 20

alpha = 0.43

if alpha == 0.43:
    # Transfer.
    dict_l_list = [
        "../results_taxi_IJCAI/alpha=0.43 using alpha=0.50 (5 seeds)/seed 1239/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 using alpha=0.50 (5 seeds)/seed 1240/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 using alpha=0.50 (5 seeds)/seed 1241/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 using alpha=0.50 (5 seeds)/seed 1242/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 using alpha=0.50 (5 seeds)/seed 1243/7499.tar",
    ]
    # Non-transfer.
    dict_r_list = [
        "../results_taxi_IJCAI/alpha=0.43 (5 seeds)/seed 1240/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 (5 seeds)/seed 1242/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 (5 seeds)/seed 1243/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 (5 seeds)/seed 1245/7499.tar",
        "../results_taxi_IJCAI/alpha=0.43 (5 seeds)/seed 1249/7499.tar",
    ]
elif alpha == 0.63:
    # Transfer.
    dict_l_list = [
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1234 (original)/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1235/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1236/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1237/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1238/7499.tar",
        ]
    # Non-transfer.
    dict_r_list = [
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1234 (original)/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1235/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1236/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1237/7499.tar",
        "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1238/7499.tar",
        ]
elif alpha == 0.70:
    # Transfer.
    dict_l_list = [
        "../results_taxi_IJCAI/alpha=0.70 using alpha=0.87 (5 seeds)/seed 1239/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 using alpha=0.87 (5 seeds)/seed 1240/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 using alpha=0.87 (5 seeds)/seed 1241/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 using alpha=0.87 (5 seeds)/seed 1242/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 using alpha=0.87 (5 seeds)/seed 1243/7499.tar",
    ]
    # Non-transfer.
    dict_r_list = [
        "../results_taxi_IJCAI/alpha=0.70 (5 seeds)/seed 1239/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 (5 seeds)/seed 1240/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 (5 seeds)/seed 1242/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 (5 seeds)/seed 1243/7499.tar",
        "../results_taxi_IJCAI/alpha=0.70 (5 seeds)/seed 1245/7499.tar",
    ]
elif alpha == 0.93:
    # Transfer.
    dict_l_list = [
        "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1238/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1239/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1240/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1241/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 using alpha=1.00 (5 seeds)/seed 1242/7499.tar",
    ]
    # Non-transfer.
    dict_r_list = [
        "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1234/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1235/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1236/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1237/7499.tar",
        "../results_taxi_IJCAI/alpha=0.93 (5 seeds)/seed 1242/7499.tar",
    ]

outcomes_l = [np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500])]  # [orr, osc, avg_rew, obj]

outcomes_r = [np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500])]


for l in range(5):
    data_l = torch.load(dict_l_list[l])
    outcomes_l_one_seed = data_l["outcomes_t"]
    for i in range(4):  # outcomes_l_one_seed is a list which consists [orr, osc, avg_rew, obj].
        outcomes_l_one_seed[i] = tile_rav_mult_idx(outcomes_l_one_seed[i], [NUM_TESTS, 7500])
        outcomes_l[i] = np.concatenate((outcomes_l[i], outcomes_l_one_seed[i]), axis=0)

for r in range(5):
    data_r = torch.load(dict_r_list[r])
    outcomes_r_one_seed = data_r["outcomes_t"]
    for i in range(4):
        outcomes_r_one_seed[i] = tile_rav_mult_idx(outcomes_r_one_seed[i], [NUM_TESTS, 7500])
        outcomes_r[i] = np.concatenate((outcomes_r[i], outcomes_r_one_seed[i]), axis=0)

font_settings = {
    'font_name': 'Times',
    'axis_size': 40,  # 24 for the single graph.
    'legend_size': 40,  # 20 for the single graph.
    'tick_size': 40,  # 20 for the single graph.
}

utils_taxi.get_plt_final_grayscale_only_obj(outcomes_l, outcomes_r, font_settings=font_settings)
