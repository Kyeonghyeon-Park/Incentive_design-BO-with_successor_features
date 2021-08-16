import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch

def save_img(rgb_arr, path, name):
    plt.imshow(rgb_arr, interpolation="nearest")
    plt.savefig(path + name)


def make_video_from_image_dir(vid_path, img_folder, video_name="trajectory", fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort()

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)


def make_video_from_rgb_imgs(
    rgb_arrs, vid_path, video_name="trajectory", fps=5, format="mp4v", resize=None
):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != "/":
        vid_path += "/"
    video_path = vid_path + video_name + ".mp4"

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, _ = frame.shape
        resize = width, height

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 20 == 0:
            print("\t...", percent_done, "% of frames rendered")
        # Always resize, without this line the video does not render properly.
        image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        video.write(image)

    video.release()


def return_view(grid, pos, row_size, col_size):
    """Given a map grid, position and view window, returns correct map part

    Note, if the agent asks for a view that exceeds the map bounds,
    it is padded with zeros

    Parameters
    ----------
    grid: 2D array
        map array containing characters representing
    pos: np.ndarray
        list consisting of row and column at which to search
    row_size: int
        how far the view should look in the row dimension
    col_size: int
        how far the view should look in the col dimension

    Returns
    -------
    view: (np.ndarray) - a slice of the map for the agent to see
    """
    x, y = pos
    left_edge = x - col_size
    right_edge = x + col_size
    top_edge = y - row_size
    bot_edge = y + row_size
    pad_mat, left_pad, top_pad = pad_if_needed(left_edge, right_edge, top_edge, bot_edge, grid)
    x += left_pad
    y += top_pad
    view = pad_mat[x - col_size : x + col_size + 1, y - row_size : y + row_size + 1]
    return view


def pad_if_needed(left_edge, right_edge, top_edge, bot_edge, matrix):
    # FIXME(ev) something is broken here, I think x and y are flipped
    row_dim = matrix.shape[0]
    col_dim = matrix.shape[1]
    left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
    if left_edge < 0:
        left_pad = abs(left_edge)
    if right_edge > row_dim - 1:
        right_pad = right_edge - (row_dim - 1)
    if top_edge < 0:
        top_pad = abs(top_edge)
    if bot_edge > col_dim - 1:
        bot_pad = bot_edge - (col_dim - 1)

    return (
        pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0),
        left_pad,
        top_pad,
    )


def pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
    pad_mat = np.pad(
        matrix,
        ((left_pad, right_pad), (top_pad, bot_pad)),
        "constant",
        constant_values=(const_val, const_val),
    )
    return pad_mat


def get_all_subdirs(path):
    return [path + "/" + d for d in os.listdir(path) if os.path.isdir(path + "/" + d)]


def get_all_files(path):
    return [path + "/" + d for d in os.listdir(path) if not os.path.isdir(path + "/" + d)]


def update_nested_dict(d0, d1):
    """
    Recursively updates a nested dictionary with a second nested dictionary.
    This function exists because the standard dict update overwrites nested dictionaries instead of
    recursively updating them.
    :param d0: The dict that receives the new values
    :param d1: The dict providing new values
    :return: Nothing, d0 is updated in place
    """
    for k, v in d1.items():
        if k in d0 and type(v) is dict:
            if type(d0[k]) is dict:
                update_nested_dict(d0[k], d1[k])
            else:
                raise TypeError
        else:
            d0[k] = d1[k]


def draw_rgb_array(rgb_array):
    """
    This function is only for check
    This function draws rgb array (it can be used for drawing the observation)
    """
    plt.cla()
    plt.imshow(rgb_array, interpolation="nearest")
    plt.show()


def draw_or_save_plt_v1(outcome, mode='draw', filename=''):
    """
    Draw or save the graph of cumulative collective rewards

    Parameters
    ----------
    outcome : list
        Outcome for previous episodes
    mode : str
        'draw' if we want to draw the figure
        'save' if we want to save the figure
    filename : str
        path name for saving the figure
    """
    plt.figure(figsize=(16, 14))

    plt.plot(outcome, label='Cumulative collective rewards')
    plt.ylim([0, np.max(outcome) + 1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    if mode == 'draw':
        plt.show()
    elif mode == 'save':
        plt.savefig(filename)
    else:
        raise ValueError


def draw_or_save_plt_v2(collective_rewards, mode='draw', filename=''):
    """
    Draw or save the graph of collective rewards

    Parameters
    ----------
    collective_rewards : numpy.ndarray
        Array of collective reward for each episode
    mode : str
        'draw' if we want to draw the figure
        'save' if we want to save the figure
    filename : str
        path name for saving the figure
    """
    plt.figure(figsize=(16, 14))
    x = np.arange(collective_rewards.size)
    plt.scatter(x, collective_rewards, label='Collective rewards')
    plt.ylim([0, np.max(collective_rewards) + 1])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Collective rewards per episode', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    if mode == 'draw':
        plt.show()
    elif mode == 'save':
        plt.savefig(filename)
    else:
        raise ValueError


def draw_or_save_plt_v3(collective_rewards, i=0, mode='draw', filename=''):
    """
    Draw or save the graph of collective rewards

    Parameters
    ----------
    collective_rewards : numpy.ndarray
        Array of collective reward for each episode
    i : int
        Number of finished episodes
    mode : str
        'draw' if we want to draw the figure
        'save' if we want to save the figure
    filename : str
        path name for saving the figure
    """

    rew = collective_rewards[:i+1]
    moving_avg_len = 20
    means = np.zeros(rew.size)
    stds = np.zeros(rew.size)
    for j in range(rew.size):
        if j + 1 < moving_avg_len:
            rew_part = rew[:j + 1]
        else:
            rew_part = rew[j - moving_avg_len + 1:j + 1]
        means[j] = np.mean(rew_part)
        stds[j] = np.std(rew_part)

    plt.figure(figsize=(16, 14))

    x = np.arange(rew.size)

    plt.plot(x, means, label='Moving avg. of collective rewards')
    plt.fill_between(x, means - stds, means + stds, color=(0.85, 0.85, 1))
    plt.scatter(x, rew, label='Collective rewards')
    plt.ylim([0, np.max(rew) + 1])
    plt.xlabel('Episodes (1000 steps per episode)', fontsize=20)
    plt.ylabel('Collective rewards per episode', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    if mode == 'draw':
        plt.show()
    elif mode == 'save':
        plt.savefig(filename)
    else:
        raise ValueError


def save_data_v1(args, env, buffers, time_trained, rewards, networks, path, name):
    """
    Function which saves several data
    """
    if args.mode_ac:
        actor_params = networks.actor.state_dict()
        actor_opt_params = networks.opt_actor.state_dict()
    else:
        actor_params = None
        actor_opt_params = None
    if args.mode_psi:
        psi_params = networks.psi.state_dict()
        psi_opt_params = networks.opt_psi.state_dict()
        critic_params = None
        critic_opt_params = None
    else:
        psi_params = None
        psi_opt_params = None
        critic_params = networks.critic.state_dict()
        critic_opt_params = networks.opt_critic.state_dict()

    torch.save({
        'args': args,
        'env': env,
        'buffers': buffers,
        'time_trained': time_trained,
        'rewards': rewards,
        'actor': actor_params,
        'psi': psi_params,
        'critic': critic_params,
        'opt_actor': actor_opt_params,
        'opt_psi': psi_opt_params,
        'opt_critic': critic_opt_params,
    }, path + name)


def save_data_v2(args, env, time_trained, collective_rewards, networks, path, name):
    actor_params = networks.actor.state_dict()
    actor_opt_params = networks.actor_opt.state_dict()
    critic_params = networks.critic.state_dict()
    critic_opt_params = networks.critic_opt.state_dict()
    torch.save({
        'args': args,
        'env': env,
        'time_trained': time_trained,
        'collective_rewards': collective_rewards,
        'actor': actor_params,
        'actor_opt': actor_opt_params,
        'critic': critic_params,
        'critic_opt': critic_opt_params,
    }, path + name)


def save_data_v3(args, env, episode_trained, decayed_eps, time_trained, collective_rewards, networks, path, name):
    """
    Function which saves several data
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

    torch.save({
        'args': args,
        'env': env,
        'episode_trained': episode_trained,
        'time_trained': time_trained,
        'decayed_eps': decayed_eps,
        'collective_rewards': collective_rewards,
        'actor': actor_params,
        'actor_opt': actor_opt_params,
        'psi': psi_params,
        'psi_opt': psi_opt_params,
        'critic': critic_params,
        'critic_opt': critic_opt_params,
    }, path + name)


def make_setting_txt(args, path):
    """
    Save current setting(args) to txt for easy check

    Parameters
    ----------
    args
        args which contains current setting
    path : str
        Path where txt file is stored
    """
    txt_path = os.path.join(path, 'args.txt')
    f = open(txt_path, 'w')
    for arg in vars(args):
        content = arg + ': ' + str(getattr(args, arg)) + '\n'
        f.write(content)
    f.close()
