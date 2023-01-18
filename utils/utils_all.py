import os
import random
import time

import numpy as np
import torch
import torch.nn as nn


def get_current_time_tag():
    """
    Return string of current time.

    Returns
    -------
    time_tag: str
    """
    time_tag = "_"+time.strftime('%y%m%d_%H%M', time.localtime(time.time()))
    return time_tag


def make_setting_txt(args, path):
    """
    Save current setting(args) to txt for easy check.

    Parameters
    ----------
    args: argparse.Namespace
        args which contains current setting.
    path: str
        Path where txt file is stored.
    """
    txt_path = os.path.join(path, 'args.txt')
    f = open(txt_path, 'w')
    for arg in vars(args):
        content = arg + ': ' + str(getattr(args, arg)) + '\n'
        f.write(content)
    f.close()


def set_random_seed(rand_seed):
    """
    Set random seeds.
    We might use np.random.RandomState() to update this function.

    Parameters
    ----------
    rand_seed: int
    """
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)


def init_weights(m):
    """
    Define the initialization function for the layers.

    Parameters
    ----------
    m
        Type of the layer.
    """
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def load_networks(networks, args, dict_trained):
    """
    Load trained networks' parameters.

    Parameters
    ----------
    networks: networks_taxi.Networks or networks_ssd.Networks
    args: argparse.Namespace
    dict_trained: dict

    Returns
    -------
    networks: networks_taxi.Networks or networks_ssd.Networks
    """
    if args.mode_ac:
        networks.actor.load_state_dict(dict_trained['actor'])
        networks.actor_target.load_state_dict(dict_trained['actor'])
    if args.mode_psi:
        networks.psi.load_state_dict(dict_trained['psi'])
        networks.psi_target.load_state_dict(dict_trained['psi'])
    else:
        networks.critic.load_state_dict(dict_trained['critic'])
        networks.critic_target.load_state_dict(dict_trained['critic'])
    return networks


def validate_setting(args):
    """
    Validate the current setting(=args).

    Parameters
    ----------
    args: argparse.Namespace
    """
    if args.mode_ac:
        assert len(args.h_dims_a) != 0 and args.lr_a != 0, "Actor network setting error."
    if args.mode_psi:
        assert len(args.h_dims_p) != 0 and args.lr_p != 0, "Psi network setting error."
    else:
        assert len(args.h_dims_c) != 0 and args.lr_c != 0, "Critic network setting error."
    if args.mode_reuse_networks:
        dict_trained = torch.load(args.file_path)
        args_trained = dict_trained['args']
        is_true = (args.mode_psi == args_trained.mode_psi) and (args.mode_ac == args_trained.mode_ac)
        assert is_true, "You can not reuse other networks which modes are not matched."


def get_networks_params(args, networks):
    """
    Get current networks' parameters.

    Parameters
    ----------
    args: argparse.Namespace
    networks: networks_taxi.Networks or networks_ssd.Networks

    Returns
    -------
    actor_params: None or collections.OrderedDict
    actor_opt_params: None or collections.OrderedDict
    critic_params: None or collections.OrderedDict
    critic_opt_params: None or collections.OrderedDict
    psi_params: None or collections.OrderedDict
    psi_opt_params: None or collections.OrderedDict
    """
    actor_params, actor_opt_params, critic_params, critic_opt_params, psi_params, psi_opt_params = [None] * 6

    if args.mode_ac:
        actor_params = networks.actor.state_dict()
        actor_opt_params = networks.actor_opt.state_dict()
    if args.mode_psi:
        psi_params = networks.psi.state_dict()
        psi_opt_params = networks.psi_opt.state_dict()
    else:
        critic_params = networks.critic.state_dict()
        critic_opt_params = networks.critic_opt.state_dict()

    return actor_params, actor_opt_params, critic_params, critic_opt_params, psi_params, psi_opt_params


def tile_ravel_multi_index(a, dims):
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
