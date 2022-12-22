import torch
from utils import utils_ssd


outcomes_names = [
    '../evaluation_results_ssd_221221_1650',
    # '../evaluation_results_ssd_221218_0116',
]

ALPHA = 0.14

for outcomes_name in outcomes_names:
    outcomes = torch.load(outcomes_name+'.tar')
    utils_ssd.get_plt_test_axis_fixed(outcomes[ALPHA][ALPHA], outcomes_name)
