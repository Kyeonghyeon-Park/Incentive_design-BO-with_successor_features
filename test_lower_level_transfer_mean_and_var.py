import numpy as np

transfer_outcomes = [4000, 5000, 2000, 9000,]
non_transfer_outcomes = [23000, 16000, 8000, 20000,]
# transfer_outcomes = [14, 11, 15, 11,]
# non_transfer_outcomes = [17, 13, 19, 14,]

mean_t = np.mean(transfer_outcomes)
mean_n = np.mean(non_transfer_outcomes)
std_t = np.std(transfer_outcomes)
std_n = np.std(non_transfer_outcomes)

print(f'Mean')
print(f'Transfer: {mean_t:.2f}, Non-transfer: {mean_n:.2f}')
print(f'Std')
print(f'Transfer: {std_t:.2f}, Non-transfer: {std_n:.2f}')