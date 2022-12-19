import torch
from utils import utils_ssd


outcomes_names = [
    '../evaluation_results_ssd_221219_1116',
    # '../evaluation_results_ssd_221218_0116',
]

for outcomes_name in outcomes_names:
    outcomes = torch.load(outcomes_name+'.tar')
    utils_ssd.get_plt_test_axis_fixed(outcomes[0.13][0.13], outcomes_name)
