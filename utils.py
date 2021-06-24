import torch
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import time


def get_actor_input(observation):
    """
    Define the actor input generation(conversion) function because of the categorical data
    [0, 1, 2, 3 : location / 4, 5, 6 : time]

    Parameters
    ----------
    observation : np.array
        (location, time)

    Returns
    -------
    actor_input : torch.Tensor
        Return the converted observation for the actor network
        Size : (1, 7)
    """
    actor_input_numpy = np.zeros(7)
    location = observation[0]
    current_time = observation[1]
    actor_input_numpy[location] = 1
    if current_time > 2:
        actor_input_numpy[6] = 1
    else:
        actor_input_numpy[4 + current_time] = 1
    actor_input = torch.FloatTensor(actor_input_numpy).unsqueeze(0)

    return actor_input


def get_critic_input(observation, action, mean_action):
    """
    Define the critic input generation(conversion) function because of the categorical data
    [0, 1, 2, 3 : location / 4, 5, 6 : time / 7, 8, 9, 10 : action / 11 : mean action]

    Parameters
    ----------
    observation : np.array
        (location, time)
    action : int
    mean_action : float

    Returns
    -------
    critic_input : torch.Tensor
        Return the converted input for the critic network
        Size : (1, 12)
    """
    critic_input_numpy = np.zeros(12)
    location = observation[0]
    current_time = observation[1]
    critic_input_numpy[location] = 1
    if current_time > 2:
        critic_input_numpy[6] = 1
    else:
        critic_input_numpy[4 + current_time] = 1
    critic_input_numpy[4] = current_time
    critic_input_numpy[7 + action] = 1
    critic_input_numpy[11] = np.min([mean_action, 1])
    critic_input = torch.FloatTensor(critic_input_numpy).unsqueeze(0)

    return critic_input


def get_psi_input(observation, action, mean_action):
    """
    Define the psi input generation function which is same as the get_critic_input
    [0, 1, 2, 3 : location / 4, 5, 6 : time / 7, 8, 9, 10 : action / 11 : mean action]

    Parameters
    ----------
    observation : np.array
    action : int
    mean_action : float

    Returns
    -------
    psi_input : torch.Tensor
        Return the converted input for the psi network (successor feature)
    """
    psi_input = get_critic_input(observation, action, mean_action)
    return psi_input


def get_q(psi, w):
    """
    Get q value using psi and w

    Parameters
    ----------
    psi : torch.Tensor
    w : numpy.array

    Returns
    -------
    q_value : float
    """
    psi = np.array(psi)
    psiT = psi.reshape(w.shape)
    q = np.dot(psiT, w)
    q_value = q.item()

    return q_value


def get_action_dist(actor_network, observation):
    """
    Define the action distribution generation function given actor network and observation (=pi_network(a_i|o_i))

    Parameters
    ----------
    actor_network : actor_critic.Actor
        Actor network
    observation : np.array
        Observation (input for the actor network)

    Returns
    -------
    action_dist : torch.distributions.categorical.Categorical
        Return the categorical distribution of the action
    """
    actor_input = get_actor_input(observation)
    action_prob = actor_network(actor_input)
    if observation[0] == 0:
        available_action_torch = torch.tensor([1, 1, 1, 0])
    elif observation[0] == 1:
        available_action_torch = torch.tensor([1, 1, 0, 1])
    elif observation[0] == 2:
        available_action_torch = torch.tensor([1, 0, 1, 1])
    else:
        available_action_torch = torch.tensor([0, 1, 1, 1])
    action_dist = distributions.Categorical(torch.mul(action_prob, available_action_torch))

    return action_dist


def check_results_in_console(file):
    """
    For easily check the previous results

    Examples
    --------
    file = 'C:/Users/ParkKH/Dropbox/KAIST/03. 연구/Reward structure design/Experiment/BO and MFRL/weights/a_lr=0.0001_alpha=0.3197/201031_1436/all_5499episode.tar'
    file = 'C:/Users/ParkKH/Dropbox/KAIST/04. 프로젝트/03. IDC/Incentive_design-BO-with_successor_features/backup (~210308_2136)/results/a_lr=0.0005_alpha=0.1/210308_1642/all_2499episode.tar'
    file = 'C:/Users/ParkKH/Dropbox/KAIST/04. 프로젝트/03. IDC/Incentive_design-BO-with_successor_features/results/alpha=0.5/210308_1642/all_2499episode.tar'
    data = check_results_in_console(file)
    outcome = data['outcome']
    draw_plt_avg(outcome, 10)

    Parameters
    ----------
    file : str
        Link of the result file

    Returns
    -------
    data : dict
        Return the results
    """
    data = torch.load(file)
    return data


def add_network_index(old_previous_networks):
    """
    To choose the network specifically (when there are several networks for same alpha),
    I add networks_index to the previous_networks
    After adding index, previous_networks is the list of [w, actor, psi, network_idx]
    You should use this function "one" time for your old version of previous_networks

    Parameters
    ----------
    old_previous_networks : list
        old_previous_networks is the list of [w, actor, psi]
    """
    idx = 0
    for networks in old_previous_networks:
        networks.append(idx)
        idx = idx + 1
    save_previous_networks(old_previous_networks)


def load_previous_networks():
    """
    For easily check and modify the previous networks in console

    Returns
    -------
    previous_networks : list
    """
    previous_networks_dict = torch.load('./weights_and_networks/previous_networks.tar')
    previous_networks = previous_networks_dict['previous_networks']
    return previous_networks


def save_previous_networks(previous_networks):
    """
    Save previous networks at the location

    Parameters
    ----------
    previous_networks : list
    """
    torch.save({
        'previous_networks': previous_networks,
    }, './weights_and_networks/previous_networks.tar')


def delete_recent_previous_networks(previous_networks):
    """
    For easily drop the last elements([w, actor, psi]) of previous networks

    Parameters
    ----------
    previous_networks : list

    Returns
    -------
    previous_networks : list
    """
    previous_networks = previous_networks[:len(previous_networks)-1]
    return previous_networks


def draw_plt(outcome):
    """
    Draw the graph of outcome (avg reward, ORR, OSC, obj. of train and test)
    Cannot be used in new version

    Parameters
    ----------
    outcome : dict
        Outcome for previous episodes
    """
    plt.figure(figsize=(16, 14))

    plt.subplot(2, 2, 1)
    plt.plot(outcome['train']['avg_reward'], label='Avg reward train')
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(outcome['test']['avg_reward'], label='Avg reward test')
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(outcome['train']['ORR'], label='ORR train')
    plt.plot(outcome['train']['OSC'], label='OSC train')
    plt.plot(outcome['train']['obj_ftn'], label='Obj train')
    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(outcome['test']['ORR'], label='ORR test')
    plt.plot(outcome['test']['OSC'], label='OSC test')
    plt.plot(outcome['test']['obj_ftn'], label='Obj test')
    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.show()


def draw_plt_avg(outcome, moving_avg_length):
    """
    Draw the moving average graph of outcome (avg reward, ORR, OSC, obj. of train and test)
    If the number of episodes are less than the window size, average all previous episodes (the number of episodes)
    Cannot be used in new version

    Parameters
    ----------
    outcome : dict
        Outcome for previous episodes
    moving_avg_length : int
        Window size for moving average
    """
    outcome_avg = {}
    for i in outcome:
        outcome_avg[i] = {}
        for j in outcome[i]:
            measure_avg = []
            for k in range(len(outcome[i][j])):
                if k < moving_avg_length - 1:
                    measure_avg.append(np.average(outcome[i][j][:k + 1]))
                else:
                    measure_avg.append(np.average(outcome[i][j][k - moving_avg_length + 1:k + 1]))
            outcome_avg[i][j] = measure_avg
    draw_plt(outcome_avg)


def draw_plt_test(outcome, episode):
    """
    Draw the graph of outcome (avg reward, ORR, OSC, obj. of train and test)

    Parameters
    ----------
    outcome : dict
        Outcome for previous episodes
    episode : int
    """

    # x축 만들기
    episode = episode + 1
    x_axis = np.linspace(0, episode, num=episode, endpoint=False)

    result = {}
    for i in outcome:
        result[i] = {}
        for j in outcome[i]:
            values = np.array(outcome[i][j])
            means = np.mean(values, axis=0)
            stds = np.std(values, axis=0)
            result[i][j] = {'mean': means,
                            'std': stds, }

    plt.figure(figsize=(16, 14))

    plt.subplot(2, 2, 1)
    mean = result['train']['avg_reward']['mean']
    std = result['train']['avg_reward']['std']
    plt.plot(x_axis, mean, label='Avg reward train', color=(0, 0, 1))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.title('Train(average reward)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    plt.subplot(2, 2, 2)
    mean = result['test']['avg_reward']['mean']
    std = result['test']['avg_reward']['std']
    plt.plot(x_axis, mean, label='Avg reward test', color=(0, 0, 1))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.title('Test(average reward)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    plt.subplot(2, 2, 3)
    mean = result['train']['ORR']['mean']
    std = result['train']['ORR']['std']
    plt.plot(x_axis, mean, label='ORR train', color=(0, 0, 1))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))
    mean = result['train']['OSC']['mean']
    std = result['train']['OSC']['std']
    plt.plot(x_axis, mean, label='OSC train', color=(1, 0, 0))
    plt.fill_between(x_axis, mean - std, mean + std, color=(1, 0.75, 0.75))
    mean = result['train']['obj_ftn']['mean']
    std = result['train']['obj_ftn']['std']
    plt.plot(x_axis, mean, label='Obj train', color=(0, 1, 0))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 1, 0.75))
    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.title('Train(objective)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    plt.subplot(2, 2, 4)
    mean = result['test']['ORR']['mean']
    std = result['test']['ORR']['std']
    plt.plot(x_axis, mean, label='ORR test', color=(0, 0, 1))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))

    mean = result['test']['OSC']['mean']
    std = result['test']['OSC']['std']
    plt.plot(x_axis, mean, label='OSC test', color=(1, 0, 0))
    plt.fill_between(x_axis, mean - std, mean + std, color=(1, 0.75, 0.75))

    mean = result['test']['obj_ftn']['mean']
    std = result['test']['obj_ftn']['std']
    plt.plot(x_axis, mean, label='Obj test', color=(0, 1, 0))
    plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 1, 0.75))

    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.title('Test(objective)', fontdict={'fontsize': 24})
    plt.legend(loc='lower right', fontsize=20)
    plt.grid()

    plt.show()


def draw_plt_two_outcome(outcome_1, outcome_2, label_1, label_2, episode):
    """
    Draw the graphs of two outcomes (avg reward, ORR, OSC, obj. of train and test)
    Not implemented yet

    Parameters
    ----------
    outcome : dict
        Outcome for previous episodes
    episode : int
    """
    raise NotImplementedError
    # # x축 만들기
    # episode = episode + 1
    # x_axis = np.linspace(0, episode, num=episode, endpoint=False)
    #
    # result = {}
    # for i in outcome:
    #     result[i] = {}
    #     for j in outcome[i]:
    #         values = np.array(outcome[i][j])
    #         means = np.mean(values, axis=0)
    #         stds = np.std(values, axis=0)
    #         result[i][j] = {'mean': means,
    #                         'std': stds, }
    #
    # plt.figure(figsize=(16, 14))
    #
    # plt.subplot(2, 2, 1)
    # mean = result['train']['avg_reward']['mean']
    # std = result['train']['avg_reward']['std']
    # plt.plot(x_axis, mean, label='Avg reward train', color=(0, 0, 1))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))
    # plt.ylim([0, 6])
    # plt.xlabel('Episode', fontsize=20)
    # plt.ylabel('Value', fontsize=20)
    # plt.title('Train(average reward)', fontdict={'fontsize': 24})
    # plt.legend(loc='lower right', fontsize=20)
    # plt.grid()
    #
    # plt.subplot(2, 2, 2)
    # mean = result['test']['avg_reward']['mean']
    # std = result['test']['avg_reward']['std']
    # plt.plot(x_axis, mean, label='Avg reward test', color=(0, 0, 1))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))
    # plt.ylim([0, 6])
    # plt.xlabel('Episode', fontsize=20)
    # plt.ylabel('Value', fontsize=20)
    # plt.title('Test(average reward)', fontdict={'fontsize': 24})
    # plt.legend(loc='lower right', fontsize=20)
    # plt.grid()
    #
    # plt.subplot(2, 2, 3)
    # mean = result['train']['ORR']['mean']
    # std = result['train']['ORR']['std']
    # plt.plot(x_axis, mean, label='ORR train', color=(0, 0, 1))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))
    # mean = result['train']['OSC']['mean']
    # std = result['train']['OSC']['std']
    # plt.plot(x_axis, mean, label='OSC train', color=(1, 0, 0))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(1, 0.75, 0.75))
    # mean = result['train']['obj_ftn']['mean']
    # std = result['train']['obj_ftn']['std']
    # plt.plot(x_axis, mean, label='Obj train', color=(0, 1, 0))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 1, 0.75))
    # plt.ylim([0, 1.1])
    # plt.xlabel('Episode', fontsize=20)
    # plt.ylabel('Value', fontsize=20)
    # plt.title('Train(objective)', fontdict={'fontsize': 24})
    # plt.legend(loc='lower right', fontsize=20)
    # plt.grid()
    #
    # plt.subplot(2, 2, 4)
    # mean = result['test']['ORR']['mean']
    # std = result['test']['ORR']['std']
    # plt.plot(x_axis, mean, label='ORR test', color=(0, 0, 1))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 0.75, 1))
    #
    # mean = result['test']['OSC']['mean']
    # std = result['test']['OSC']['std']
    # plt.plot(x_axis, mean, label='OSC test', color=(1, 0, 0))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(1, 0.75, 0.75))
    #
    # mean = result['test']['obj_ftn']['mean']
    # std = result['test']['obj_ftn']['std']
    # plt.plot(x_axis, mean, label='Obj test', color=(0, 1, 0))
    # plt.fill_between(x_axis, mean - std, mean + std, color=(0.75, 1, 0.75))
    #
    # plt.ylim([0, 1.1])
    # plt.xlabel('Episode', fontsize=20)
    # plt.ylabel('Value', fontsize=20)
    # plt.title('Test(objective)', fontdict={'fontsize': 24})
    # plt.legend(loc='lower right', fontsize=20)
    # plt.grid()
    #
    # plt.show()


def print_updated_q(critic):
    """
    Print the q value for all locations, times, actions and some mean actions
    To do : get the value of the number of locations and the episode length from the world setting

    Parameters
    ----------
    critic : actor_critic.Critic
        Critic network
    """
    for location in range(4):
        for agent_time in range(3):
            print("Q at (#", location, ", ", agent_time, ")")
            for action in range(4):
                q = []
                for mean_action in np.arange(0.0, 1.1, 0.1):
                    critic_input = get_critic_input([location, agent_time], action, mean_action)
                    q_value = critic(critic_input)
                    q.append(q_value.item())
                q = np.array(q)
                with np.printoptions(formatter={'float': '{: 0.2f}'.format}, sign=' ', linewidth=np.inf):
                    print(q)


def print_updated_q_using_psi(psi_network, designer_alpha):
    """
    Using psi(successor feature network),
    print the q value for all locations, times, actions and some mean actions

    Parameters
    ----------
    psi_network : actor_psi.Psi
        Successor feature network
    designer_alpha : float
        Designer's decision (or penalty level)
    """
    w = np.array([1, designer_alpha])

    for location in range(4):
        for agent_time in range(3):
            print("Q(=psi*w) at (#", location, ", ", agent_time, ")")
            for action in range(4):
                q = []
                for mean_action in np.arange(0.0, 1.1, 0.1):
                    psi_input = get_psi_input([location, agent_time], action, mean_action)
                    psi = psi_network(psi_input)
                    q_value = get_q(psi, w)
                    q.append(q_value)
                q = np.array(q)
                with np.printoptions(formatter={'float': '{: 0.2f}'.format}, sign=' ', linewidth=np.inf):
                    print(q)


def print_action_distribution(actor):
    """
    Print the action distribution for all locations and times
    To do : get the value of the number of locations and the episode length from the world setting

    Parameters
    ----------
    actor : actor_critic.Actor
        Actor network
    """
    with np.printoptions(formatter={'float': '{: 0.2f}'.format}, sign=' ', linewidth=np.inf):
        for location in range(4):
            for agent_time in range(3):
                action_dist = get_action_dist(actor, [location, agent_time])
                print("Action distribution at (#", location, ", ", agent_time, ") : ", action_dist.probs[0].numpy())


def print_information_per_n_episodes(outcome, overall_time, episode):
    """
    Print the outcome of learned network per n episodes
    Printed outcome of test will be the mean of results

    Parameters
    ----------
    outcome : dict
        Outcome for previous episodes
    overall_time : dict
        Overall time for previous episodes
    episode : int
        Current episode
    """
    train_time = np.sum(overall_time['train'][:episode])
    test_time = np.sum(overall_time['test'][:episode])
    total_time = np.sum(overall_time['total'][:episode])

    result = {}
    for i in outcome:
        result[i] = {}
        for j in outcome[i]:
            values = [item[episode] for item in outcome[i][j]]
            result[i][j] = np.mean(values)
    print("############################################################################################")
    print(f"| Episode : {episode:4} "
          f"| train time : {train_time:5.2f} | test time : {test_time:5.2f} | total time : {total_time:5.2f} |")
    print(
        f"| train ORR : {result['train']['ORR']:5.4f} "
        f"| train OSC : {result['train']['OSC']:5.4f} "
        f"| train avg reward : {result['train']['avg_reward']:5.4f} "
        f"| train Obj : {result['train']['obj_ftn']:5.4f} |")
    print(
        f"|  test ORR : {result['test']['ORR']:5.4f} "
        f"|  test OSC : {result['test']['OSC']:5.4f} "
        f"|  test avg reward : {result['test']['avg_reward']:5.4f} "
        f"|  test Obj : {result['test']['obj_ftn']:5.4f} |")
    print("############################################################################################")