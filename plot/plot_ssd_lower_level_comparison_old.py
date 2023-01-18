import numpy as np
import torch

from utils import utils_ssd


''' Code for getting the lower-level transfer figure. '''
# Transfer.
dict_l_list = ["./results/211008 submitted version/results_ssd_final/alpha=0.33 using alpha=0.50 (2 seeds)/seed 1278 (original)/outcomes.tar",
               "./results/211008 submitted version/results_ssd_final/alpha=0.33 using alpha=0.50 (2 seeds)/seed 1279/outcomes.tar",
               ]
# Non-transfer.
dict_r_list = ["./results/211008 submitted version/results_ssd_final/alpha=0.33 (5 seeds)/seed 1267 (original)/outcomes.tar",
               "./results/211008 submitted version/results_ssd_final/alpha=0.33 (5 seeds)/seed 1269/outcomes.tar",
               "./results/211008 submitted version/results_ssd_final/alpha=0.33 (5 seeds)/seed 1270/outcomes.tar"
               ]
# 31 or 30.
num_networks = 30
outcomes_l = np.empty([num_networks, 0])
outcomes_r = np.empty([num_networks, 0])

# Collect and concatenate outcomes of the transfer results.
for l in dict_l_list:
    dict_l = torch.load(l)
    outcomes_l_one_seed = dict_l["obj_full"]
    outcomes_l = np.concatenate((outcomes_l, outcomes_l_one_seed), axis=1)
# Collect and concatenate outcomes of the non-transfer results.
for r in dict_r_list:
    dict_r = torch.load(r)
    outcomes_r_one_seed = dict_r["obj_full"]
    outcomes_r = np.concatenate((outcomes_r, outcomes_r_one_seed), axis=1)


font_settings = {
    'font_name': 'Times',
    'axis_size': 40,  # 24 for the single graph.
    'legend_size': 40,  # 20 for the single graph.
    'tick_size': 40,  # 20 for the single graph.
}

utils_ssd.get_plt_final_aggregate_grayscale(outcomes_l, outcomes_r, is_3000=False, font_settings=font_settings)