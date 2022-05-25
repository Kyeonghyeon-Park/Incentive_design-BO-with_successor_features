from sequential_social_dilemma_games.social_dilemmas.envs.cleanup import CleanupEnv, CleanupEnvModified
from sequential_social_dilemma_games.social_dilemmas.envs.harvest import HarvestEnv, HarvestEnvModified
from sequential_social_dilemma_games.social_dilemmas.envs.switch import SwitchEnv
from sequential_social_dilemma_games.social_dilemmas.maps import HARVEST_MAP_V2, CLEANUP_MAP_V2

def get_env_creator(env, num_agents, args):
    if env == "harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )

    elif env == "harvest_modified":

        def env_creator(_):
            return HarvestEnvModified(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                lv_penalty=args.lv_penalty,
                lv_incentive=args.lv_incentive,
            )

    elif env == "harvest_modified_v2":

        def env_creator(_):
            return HarvestEnvModified(
                ascii_map=HARVEST_MAP_V2,
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                lv_penalty=args.lv_penalty,
                lv_incentive=args.lv_incentive,
            )

    elif env == "cleanup":

        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
            )

    elif env == "cleanup_modified":

        def env_creator(_):
            return CleanupEnvModified(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                lv_penalty=args.lv_penalty,
                lv_incentive=args.lv_incentive,
            )

    elif env == "cleanup_modified_v2":

        def env_creator(_):
            return CleanupEnvModified(
                ascii_map=CLEANUP_MAP_V2,
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=args.use_collective_reward,
                lv_penalty=args.lv_penalty,
                lv_incentive=args.lv_incentive,
            )

    elif env == "switch":

        def env_creator(_):
            return SwitchEnv(num_agents=num_agents, args=args)

    return env_creator
