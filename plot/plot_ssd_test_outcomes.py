import numpy as np
import torch
from utils import utils_ssd


outcomes_names = [
    '../evaluation_results_ssd_221225_2033',
    # '../evaluation_results_ssd_221218_0116',
]

ALPHA = 0.25

for outcomes_name in outcomes_names:
    print(f'File name: {outcomes_name}')
    outcomes = torch.load(outcomes_name+'.tar')
    # Plot.
    utils_ssd.get_plt_test_axis_fixed(outcomes[ALPHA][ALPHA], outcomes_name)
    # Print the last objective value.
    print(f'Obj. value: {np.mean(outcomes[ALPHA][ALPHA]["obj"][:, -1])}')
