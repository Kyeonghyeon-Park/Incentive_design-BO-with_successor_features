import copy

import numpy as np
import torch

from main_taxi import roll_out
from networks_taxi import Networks
from taxi import TaxiEnv
from . import utils_all
"""
220504 This py file is added to resolve circular import issue.
"""


def get_env_and_networks(args, dict_trained):
    """

    Parameters
    ----------
    args: argparse.Namespace
    dict_trained: dict

    Returns
    -------
    env: TaxiEnv
    networks: Networks
    """
    env = TaxiEnv(args)
    networks = Networks(env, args)
    networks = utils_all.load_networks(networks, args, dict_trained)
    return env, networks


def get_approximated_gradient(network_path, w, w_bound, h=0.01, num_tests=100):
    """
    This function is for getting approximated gradient by calculating f(policy, w + h*e_i) - f(policy, w) / h.
    It will require several evaluations to get the gradient.

    Examples
    ----------
    import numpy as np
    from utils import funcs_taxi

    file_path = "./results/211008 submitted version/results_taxi_final/alpha=0.80/7499.tar"
    w = np.array([0.80])
    w_bound = np.array([[0, 1]])
    w_grad = funcs_taxi.get_approximated_gradient(file_path, w, w_bound, h=0.01, num_tests=1000)

    Parameters
    ----------
    network_path: str
        File path for the trained network
    w: numpy.ndarray
        Weight (or reward parameter)
        ex. w = np.array([0.33, 0.5])
    w_bound: numpy.ndarray
        List which contains bounds of w.
        Each row (call w_bound[i,:]) represents the bound of w[i].
        w[i] cannot over the bound.
        It will be used for getting gradients.
        For example, let w_bound = np.array([[0, 1]]) and w = np.array([1]).
        In this case, we can't calculate f(policy, w + h*e_i).
        It will get approximated gradient by calculating f(policy, w) - f(policy, w - h*e_i) / h.
    h: float
        Parameter for calculating approximated gradient (small value).
    num_tests: int
        The number of tests to calculate approximated gradients.
        This function evaluate "num_tests" times to get gradients and average them.

    Returns
    -------
    w_grad: numpy.ndarray
    """
    utils_all.set_random_seed(1234)
    w_grad = np.zeros(w.size)

    # Load trained data.
    dict_trained = torch.load(network_path)
    args = dict_trained['args']
    if args.lv_penalty != w:
        raise ValueError("args.lv_penalty and w are not the same. Please check your inputs.")

    # Calculate w_grad for each dimension.
    for i in range(w.size):
        # Try to build f(policy, w_f) - f(policy, w_b) / h (w_f: w_front, w_b: w_back)
        args_f, args_b = [copy.deepcopy(args) for _ in range(2)]
        if w[i] + h > w_bound[i, 1]:
            w_f, w_b = w, w - h * np.eye(w.size)[i]
        else:
            w_f, w_b = w + h * np.eye(w.size)[i], w
        args_f.lv_penalty = w_f
        args_b.lv_penalty = w_b

        # Build the environment and networks.
        env_f, networks_f = get_env_and_networks(args_f, dict_trained)
        env_b, networks_b = get_env_and_networks(args_b, dict_trained)

        # Build array for collecting objective values.
        obj_f, obj_b = [np.zeros(num_tests) for _ in range(2)]

        for j in range(num_tests):
            _, outcome_f = roll_out(networks=networks_f,
                                    env=env_f,
                                    decayed_eps=0,
                                    is_train=False)
            _, outcome_b = roll_out(networks=networks_b,
                                    env=env_b,
                                    decayed_eps=0,
                                    is_train=False)

            _, _, _, obj_f[j] = outcome_f
            _, _, _, obj_b[j] = outcome_b
            print(f"Dim: {i+1}/{w.size}, Tests: {j+1}/{num_tests}") if (((j+1) * 10) % num_tests == 0) else None

        w_grad[i] = (np.mean(obj_f) - np.mean(obj_b)) / h
        print(f"w_grad: {w_grad}")

    return w_grad
