import os
import random

import numpy as np
import torch


def make_setting_txt(args, path):
    """
    Save current setting(args) to txt for easy check.

    Parameters
    ----------
    args
        args which contains current setting.
    path : str
        Path where txt file is stored.
    """
    txt_path = os.path.join(path, 'args.txt')
    f = open(txt_path, 'w')
    for arg in vars(args):
        content = arg + ': ' + str(getattr(args, arg)) + '\n'
        f.write(content)
    f.close()


def set_random_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)


def load_networks(networks, args, dict_trained):
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