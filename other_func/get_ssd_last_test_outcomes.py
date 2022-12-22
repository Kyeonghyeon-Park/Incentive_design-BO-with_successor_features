import torch
import numpy as np


'''
221219) This code is to get the final objective value from test outcomes (of 31 network files).
'''
FILE_PATH = '../results_ssd_IJCAI/alpha=0.14 using alpha=0.26 (1 seed)/seed 1267/evaluation_results_ssd.tar'
ALPHA = 0.14

data = torch.load(FILE_PATH)
print(f'{np.mean(data[ALPHA][ALPHA]["obj"][:, -1])}')
