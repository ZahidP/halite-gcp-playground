import sys
import numpy as np

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Observation
from halite_env import HaliteEnv
from ship_state_wrapper import ShipStateWrapper
from shipyard_state_wrapper import ShipYardStateWrapper
from agent import Agent


# https://www.kaggle.com/garethjns/deep-q-learner-starter-code#Play-single-episode
# https://www.kaggle.com/hsperr/halite-iv-dqn-example-pytorch


def train_agent(episodes=20, steps_per_episode=50):
    env = make("halite", debug=True)
    # env.run(["random"])
    # env.render(mode="ipython", width=800, height=600)

    print('configuration')
    print(env.configuration)

    ship_state_wrapper = ShipStateWrapper(
        radius=4,
        max_frames=1,
        map_size=int(env.configuration['size'])
    )

    shipyard_state_wrapper = ShipYardStateWrapper(
        radius=4,
        max_frames=1,
        map_size=int(env.configuration['size'])
    )

    print(env.configuration)

    print("Initialized state wrappers")

    ship_agent = Agent(
        alpha=0.99, gamma=0.75, n_actions=6,
        batch_size=64, epsilon=.9, input_dims=ship_state_wrapper.state_size
    )

    shipyard_agent = Agent(
        alpha=0.99, gamma=0.75, n_actions=2,
        batch_size=64, epsilon=.9, input_dims=shipyard_state_wrapper.state_size
    )

    print("Initialized agents")

    trainer = env.train([None, "random", "random", "random"])
    observation = trainer.reset()

    print("Initialized trainer")

    halite_env = HaliteEnv(
        opponents=3,
        ship_state_wrapper=ship_state_wrapper,
        shipyard_state_wrapper=shipyard_state_wrapper,
        radius=4,
        trainer=trainer
    )

    all_rewards = []

    for i in range(episodes):
        episode_rewards = play_episode(
            ship_agent=ship_agent,
            shipyard_agent=shipyard_agent,
            env=halite_env,
            configuration=env.configuration,
            episode_number=i,
            training=True,
            n_steps=steps_per_episode
        )
        all_rewards.append(episode_rewards)

    return all_rewards


def play_episode(
        env: HaliteEnv,
        ship_agent: Agent,
        shipyard_agent: Agent,
        configuration,
        n_steps: int = 10,
        verbose: bool = True,
        training: bool = False,
        simulated_step_learning: bool = False,
        episode_number=0):
    episode_rewards = []
    episode_actions = []

    episode_scores = []

    raw_observation: dict = env.reset()[0].__dict__
    print('ep: {}'.format(episode_number))
    done = False

    for step_num in range(n_steps):
        if done:
            board = Board(raw_observation, raw_configuration=configuration)
            print('Done')
            print(board)
            return episode_scores

        actions_for_step = {}

        # wont change
        step_observation = Observation(raw_observation)

        shipyard_temporary_initial_memory = {}
        ship_temporary_initial_memory = {}

        """
        ====================================
        ====================================
        SHIPYARDS
        ====================================
        ====================================
        """
        for shipyard_id, (pos) in step_observation.players[step_observation.player][1].items():
            # will change at each simulated step
            board = Board(raw_observation, raw_configuration=configuration)
            observation = Observation(raw_observation)

            # Select action
            converted_observation, is_occupied = env.wrap_observation_for_shipyard_agent(
                obs=observation, player=observation.player, spos=pos, uid=shipyard_id
            )
            state_vector = converted_observation.flatten()
            state_vector: np.ndarray = np.append(state_vector, is_occupied)

            action = shipyard_agent.get_action(
                state_vector, step=step_num, game=episode_number
            )
            halite_action = env.convert_shipyard_action_to_halite_enum(action, shipyard_id, observation)
            episode_actions.append(halite_action)

            # re-aligning action and halite action
            # TODO: should refactor
            if halite_action == ShipyardAction.SPAWN:
                action = 1
            else:
                action = 0

            if halite_action:
                actions_for_step[shipyard_id] = halite_action.name

            """
            ============
            Take Action
            ============
            """
            prev_obs = observation
            obs_next = env.update_observation_for_shipyard(board, shipyard_id, halite_action)

            reward = env.get_shipyard_count_reward(
                obs_next,
                env.wrap_observation_for_ship_agent(
                    obs=obs_next,
                    player=obs_next.player,
                    spos=pos,  # because shipyards can't move
                    uid=shipyard_id
                )
            )

            episode_rewards.append(reward)

            """
            ============
            Update Model
            ============
            """

            converted_next_obs, is_occupied_next = env.wrap_observation_for_shipyard_agent(
                obs_next,
                obs_next.player,
                spos=pos,
                uid=shipyard_id
            )
            next_state_vector = converted_next_obs.flatten()
            next_state_vector: np.ndarray = np.append(next_state_vector, is_occupied_next)

            if training:
                if simulated_step_learning:
                    shipyard_agent.remember(state=state_vector, action=action,
                                            reward=reward, new_state=next_state_vector, done=done)
                    shipyard_agent.learn(step_num=step_num, episode_num=episode_number)
                else:
                    shipyard_temporary_initial_memory[shipyard_id] = {
                        'state': state_vector, 'action': action, 'pos': pos, 'is_occupied': is_occupied
                    }

            if verbose and ((n_steps % 5) == 0):
                print(f"Step {step_num}: Action taken {action} for shipyard {shipyard_id}, "
                      f"reward received {reward}")
            # update current observation with the simulated step ahead
            raw_observation = obs_next
            observation = Observation(raw_observation)

        """
        ====================================
        ====================================
        SHIPS
        ====================================
        ====================================
        """
        for ship_id, (pos, halite) in step_observation.players[step_observation.player][2].items():
            # will change at each simulated step
            board = Board(raw_observation, raw_configuration=configuration)
            observation = Observation(raw_observation)

            """
            ============
            Take Action
            ============
            """
            converted_observation = env.wrap_observation_for_ship_agent(
                obs=Observation(board.observation),
                player=board.observation['player'],
                spos=int(pos),
                uid=ship_id
            )
            state_vector = converted_observation.flatten()
            action = ship_agent.get_action(
                state_vector, step=step_num, game=episode_number
            )
            episode_actions.append(action)

            halite_action = env.convert_ship_action_to_halite_enum(action, observation)

            if halite_action and halite_action.name == halite_action.CONVERT.name and \
                observation.players[observation.player][0] < 500:
                # tried to convert without enough halite
                halite_action = None
                action = 5

            if halite_action:
                actions_for_step[ship_id] = halite_action.name

            # Take action
            prev_obs = observation
            try:
                obs_next: Observation = env.update_observation_for_ship(board, ship_id, halite_action)
            except KeyError as e:
                print('Actions taken')
                print(actions_for_step)
                print('Current board and observation')
                print(board.ships.keys())
                print(observation.players[observation.player])
                print('Initial board and observation')
                print(step_observation.players[step_observation.player])
                raise e

            # the ship may no longer exist...
            # ie it collided with an enemy ship or converted to a shipyard, we need to use the previous
            # for now we will use the new position IF it exists, otherwise just use the old one
            next_pos = obs_next.players[observation.player][2].get(ship_id, (None, None))[0]

            if not next_pos:
                next_pos = int(pos)

            reward = env.get_collector_ship_reward(
                obs_next,
                env.wrap_observation_for_ship_agent(
                    obs=obs_next,
                    player=obs_next.player,
                    spos=pos,  # because shipyards can't move
                    uid=ship_id
                ),
                ship_id
            )

            episode_rewards.append(reward)

            """
            ============
            Update Model
            ============
            """

            converted_next_obs = env.wrap_observation_for_ship_agent(obs=obs_next,
                                                                     player=obs_next.player,
                                                                     spos=next_pos,
                                                                     uid=ship_id)
            next_state_vector = converted_next_obs.flatten()

            # Update model
            if training:
                if simulated_step_learning:
                    ship_agent.remember(state=state_vector, action=action,
                                        reward=reward, new_state=next_state_vector, done=done)
                    ship_agent.learn(step_num=step_num, episode_num=episode_number)
                else:
                    ship_temporary_initial_memory[ship_id] = {'state': state_vector, 'action': action, 'pos': pos}
            action_string = halite_action.name if halite_action else 'None'

            if verbose and ((n_steps % 5) == 0):
                print(f"Step {step_num}: Action taken {action} | {action_string} for ship {ship_id}, "
                      f"reward received {reward}")
            # update current observation with the simulated step ahead
            raw_observation = obs_next

        """
        ================        
        ================
        == Take Step
        ================
        ================
        """

        # updates the env.observation
        step_results = env.step(actions=actions_for_step)

        print('Actions for step')
        print(actions_for_step)

        observation, game_reward, terminal = step_results

        if not simulated_step_learning:
            """
            Here we are doing learning after the actual "step" has taken place.

            This means that the earlier a ship or shipyard has selected its move, 
            the more unknowns and more "friendly reactions" that can occur afterwards.

            It would probably be very useful to include 
                - remaining_ship_actions
                - remaining_shipyard_actions
                - and potentially the current epsilon value
            as a part of the state.
            """

            player_halite = observation.players[observation.player][0]
            opponent_halites = [item[0] for item in observation.players[observation.player:]]
            best_opponent_halite = sorted(opponent_halites, reverse=True)[0]

            for ship_id, val in ship_temporary_initial_memory.items():
                s = val['state']
                a = val['action']
                pos = val['pos']
                converted_next_obs = env.wrap_observation_for_ship_agent(
                    obs=Observation(observation),
                    player=observation['player'],
                    spos=int(pos),
                    uid=ship_id
                )
                ship_reward = env.get_collector_ship_reward(
                    observation=observation,
                    converted_observation=converted_next_obs,
                    uid=ship_id,
                    done=done
                )
                next_state_vector = converted_next_obs.flatten()
                ship_agent.remember(state=s, action=a,
                                    reward=ship_reward, new_state=next_state_vector, done=done)
                ship_agent.learn(step_num=step_num, episode_num=episode_number)

            for shipyard_id, val in shipyard_temporary_initial_memory.items():
                s = val['state']
                a = val['action']
                pos = val['pos']
                is_occupied = val['is_occupied']
                converted_next_obs, is_occupied_next = env.wrap_observation_for_shipyard_agent(
                    obs=Observation(observation),
                    player=observation['player'],
                    spos=int(pos),
                    uid=shipyard_id
                )
                print('For action: {}'.format(a))
                shipyard_reward = env.get_shipyard_count_reward(
                    observation=observation,
                    converted_observation=converted_next_obs,
                    uid=shipyard_id,
                    done=done
                )
                next_state_vector = converted_next_obs.flatten()
                next_state_vector: np.ndarray = np.append(next_state_vector, is_occupied_next)

                shipyard_agent.remember(
                    state=s,
                    action=a,
                    reward=shipyard_reward,
                    new_state=next_state_vector,
                    done=done
                )
                shipyard_agent.learn(step_num=step_num, episode_num=episode_number)

        episode_scores.append([item[0] for item in observation['players']])
        raw_observation = observation

    return episode_scores
