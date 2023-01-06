import numpy as np
import torch

from utils import utils_ssd

''' 
Code for getting the lower-level transfer figure. 
'''
# alpha value.
alpha = 0.05

if alpha == 0.05:
    # Transfer.
    dict_l_list = [
        "../results_ssd_IJCAI/alpha=0.05 using alpha=0.00 or alpha=0.08 (3 seeds)/seed 1234 (using alpha=0.00)/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.05 using alpha=0.00 or alpha=0.08 (3 seeds)/seed 1267 (using alpha=0.08)/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.05 using alpha=0.00 or alpha=0.08 (3 seeds)/seed 1268 (using alpha=0.00)/evaluation_results_ssd.tar",
    ]

    # Non-transfer.
    dict_r_list = [
        "../results_ssd_IJCAI/alpha=0.05 (3 (of 6) seeds)/seed 1235/evaluation_results_ssd.tar",
        # "../results_ssd_IJCAI/alpha=0.05 (3 (of 6) seeds)/seed 1236/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.05 (3 (of 6) seeds)/seed 1239/evaluation_results_ssd.tar",
    ]
elif alpha == 0.33:  # Not work
    # Transfer.
    dict_l_list = [
        "../results/211008 submitted version/results_ssd_final/alpha=0.33 using alpha=0.50 (2 seeds)/seed 1278 (original)/outcomes.tar",
        "../results/211008 submitted version/results_ssd_final/alpha=0.33 using alpha=0.50 (2 seeds)/seed 1279/outcomes.tar",
    ]
    # Non-transfer.
    dict_r_list = [
        "../results/211008 submitted version/results_ssd_final/alpha=0.33 (5 seeds)/seed 1267 (original)/outcomes.tar",
        "../results/211008 submitted version/results_ssd_final/alpha=0.33 (5 seeds)/seed 1269/outcomes.tar",
        "../results/211008 submitted version/results_ssd_final/alpha=0.33 (5 seeds)/seed 1270/outcomes.tar",
    ]
elif alpha == 0.40:
    # Transfer.
    dict_l_list = [
        "../results_ssd_IJCAI/alpha=0.40 using alpha=0.33 (2 seeds)/seed 1267/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.40 using alpha=0.33 (2 seeds)/seed 1268/evaluation_results_ssd.tar",
    ]

    # Non-transfer.
    dict_r_list = [
        # "../results_ssd_IJCAI/alpha=0.40 (3 (of 9) seeds)/seed 1236 (converge to 200)/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.40 (3 (of 9) seeds)/seed 1239 (converge to 150)/evaluation_results_ssd.tar",
        # "../results_ssd_IJCAI/alpha=0.40 (3 (of 9) seeds)/seed 1240 (converge x)/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.40 (3 (of 9) seeds)/seed 1241/evaluation_results_ssd.tar",
        # "../results_ssd_IJCAI/alpha=0.40 (3 (of 9) seeds)/seed 1242/evaluation_results_ssd.tar",
    ]
elif alpha == 0.56:
    # Transfer.
    dict_l_list = [
        "../results_ssd_IJCAI/alpha=0.56 using alpha=0.33 (6 seeds)/seed 1234/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.56 using alpha=0.33 (6 seeds)/seed 1235/evaluation_results_ssd.tar",
        # "../results_ssd_IJCAI/alpha=0.56 using alpha=0.33 (6 seeds)/seed 1236/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.56 using alpha=0.33 (6 seeds)/seed 1237/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.56 using alpha=0.33 (6 seeds)/seed 1267/evaluation_results_ssd.tar",
    ]

    # Non-transfer.
    dict_r_list = [
        "../results_ssd_IJCAI/alpha=0.56 (1 (of 5) seeds, 2 evaluations needed)/seed 1234 (converge x)/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.56 (1 (of 5) seeds, 2 evaluations needed)/seed 1235 (converge x)/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=0.56 (1 (of 5) seeds, 2 evaluations needed)/seed 1236 (converge to 180)/evaluation_results_ssd.tar",
    ]
elif alpha == 1.00:
    # Transfer.
    dict_l_list = [
        "../results_ssd_IJCAI/alpha=1.00 using alpha=0.33 (3 seeds)/seed 1234/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=1.00 using alpha=0.33 (3 seeds)/seed 1235/evaluation_results_ssd.tar",
        # "../results_ssd_IJCAI/alpha=1.00 using alpha=0.33 (3 seeds)/seed 1236/evaluation_results_ssd.tar",
    ]
    # Non-transfer.
    dict_r_list = [
        "../results_ssd_IJCAI/alpha=1.00 (4 seeds)/seed 1234/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=1.00 (4 seeds)/seed 1235/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=1.00 (4 seeds)/seed 1236/evaluation_results_ssd.tar",
        "../results_ssd_IJCAI/alpha=1.00 (4 seeds)/seed 1237/evaluation_results_ssd.tar",
    ]


# 31 or 30.
num_networks = 31
outcomes_l = np.empty([0, num_networks])
outcomes_r = np.empty([0, num_networks])

# Collect and concatenate outcomes of the transfer results.
for l in dict_l_list:
    dict_l = torch.load(l)
    outcomes_l_one_seed = dict_l[alpha][alpha]["obj"]
    outcomes_l = np.concatenate((outcomes_l, outcomes_l_one_seed), axis=0)

# Collect and concatenate outcomes of the non-transfer results.
for r in dict_r_list:
    dict_r = torch.load(r)
    outcomes_r_one_seed = dict_r[alpha][alpha]["obj"]
    outcomes_r = np.concatenate((outcomes_r, outcomes_r_one_seed), axis=0)

outcomes_l_transpose = np.transpose(outcomes_l)
outcomes_r_transpose = np.transpose(outcomes_r)

font_settings = {
    'font_name': 'Times',
    'axis_size': 40,  # 24 for the single graph.
    'legend_size': 40,  # 20 for the single graph.
    'tick_size': 40,  # 20 for the single graph.
}

utils_ssd.get_plt_final_aggregate_grayscale_v3(outcomes_l_transpose, outcomes_r_transpose, font_settings=font_settings)

# Print average outcomes
print(f'Transfer outcome    : {np.mean(outcomes_l_transpose[-1, :]):.2f}')
print(f'Non-transfer outcome: {np.mean(outcomes_r_transpose[-1, :]):.2f}')
