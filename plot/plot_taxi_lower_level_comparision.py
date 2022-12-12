import numpy as np
import torch

from utils import utils_taxi


''' 
Code for getting the lower-level transfer figure. 
num_tests = 1인 tar의 result는 duplicate하여 갯수를 맞춤. 
주의사항으로, ssd에서 저장하는 outcome과 row, column이 반대임.
'''
# Transfer.
dict_l_list = ["../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1234 (original)/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1235/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1236/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1237/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 using alpha=0.50 (5 seeds)/seed 1238/7499.tar",
               ]
# Non-transfer.
dict_r_list = ["../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1234 (original)/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1235/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1236/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1237/7499.tar",
               "../results/211008 submitted version/results_taxi_final/alpha=0.63 (5 seeds)/seed 1238/7499.tar",
               ]

outcomes_l = [np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500])]

outcomes_r = [np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500]),
              np.empty([0, 7500])]

for l in range(5):
    data_l = torch.load(dict_l_list[l])
    outcomes_l_one_seed = data_l["outcomes_t"]
    for i in range(4):
        if l != 0:
            outcomes_l_one_seed[i] = np.tile(outcomes_l_one_seed[i], (20, 1))
        outcomes_l[i] = np.concatenate((outcomes_l[i], outcomes_l_one_seed[i]), axis=0)

for r in range(5):
    data_r = torch.load(dict_r_list[r])
    outcomes_r_one_seed = data_r["outcomes_t"]
    for i in range(4):
        if r != 0:
            outcomes_r_one_seed[i] = np.tile(outcomes_r_one_seed[i], (20, 1))
        outcomes_r[i] = np.concatenate((outcomes_r[i], outcomes_r_one_seed[i]), axis=0)

utils_taxi.get_plt_final_grayscale_only_obj(outcomes_l, outcomes_r)
