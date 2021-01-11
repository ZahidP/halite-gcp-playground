import sys
import numpy as np

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Observation
from halite_env import HaliteEnv
from ship_state_wrapper import ShipStateWrapper
from shipyard_state_wrapper import ShipYardStateWrapper
from agent import Agent


class Game:

    def __init__(self, env,
                 shipyard_agent,
                 ship_agent,
                 training,
                 configuration,
                 max_steps=100,
                 verbose=True,
                 ship_frame_stack_len=1,
                 shipyard_frame_stack_len=1,
                 handicap=0
                 ):
        # code smell...too many properties here
        self.actions = {}
        self.env: HaliteEnv = env
        self.shipyard_agent = shipyard_agent
        self.ship_agent = ship_agent
        self.episode_number = 0
        self.step_number = 0
        self.episode_actions = []
        self.episode_rewards = []
        self.training = training
        self.configuration = configuration
        self.verbose = verbose
        self.actions_for_step = {}
        self.max_steps = max_steps
        self.episode_score = 0
        self.shipyard_step_memory = {}
        self.ship_step_memory = {}
        self.ship_frame_stack_len = ship_frame_stack_len
        self.shipyard_frame_stack_len = shipyard_frame_stack_len
        self.handicap = handicap

    def play_episode(self, max_steps):
        raw_observation: dict = self.env.reset()[0].__dict__
        raw_observation[
            'players'
        ][raw_observation['player']][0] = raw_observation['players'][raw_observation['player']][0] - self.handicap

        step_observation = Observation(raw_observation)
        episode_score = []

        print("Episode start")
        print(raw_observation['players'])

        for step in range(max_steps):
            self.step_number = step
            step_scores = self.play_step(
                raw_observation=raw_observation,
                step_observation=step_observation
            )
            episode_score.append(step_scores)
        self.episode_number += 1
        return episode_score

    def play_step(self, raw_observation, step_observation):
        self.actions_for_step = {}
        self.shipyard_step_memory = {}
        self.ship_step_memory = {}

        self.step_number += 1

        raw_observation, shipyard_simulated_step_memory = self.get_moves_for_all_shipyards(
            raw_observation=raw_observation,
            step_observation=step_observation
        )

        raw_observation, ship_simulated_step_memory = self.get_moves_for_all_ships(
            raw_observation=raw_observation,
            step_observation=step_observation
        )

        env = self.env
        actions_for_step = self.actions_for_step

        # updates the env.observation
        step_results = env.step(actions=actions_for_step)

        print('Actions for step')
        print(actions_for_step)

        observation, game_reward, terminal = step_results

        if self.training:
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
            for ship_id, val in ship_simulated_step_memory.items():
                s = val['state']
                a = val['action']
                pos = val['pos']

                if self.step_number >= self.ship_frame_stack_len:

                    if self.ship_frame_stack_len > 1:
                        multiframe_state = self.env.get_multiframe_ship_observation(ship_id)
                        converted_obs = np.concatenate(multiframe_state, axis=0)
                        s = converted_obs.flatten()

                    converted_next_obs = env.wrap_observation_for_ship_agent(
                        obs=Observation(observation),
                        player=observation['player'],
                        spos=int(pos),
                        uid=ship_id
                    )

                    if self.ship_frame_stack_len > 1:
                        multiframe_state = self.env.get_multiframe_ship_observation(ship_id)
                        converted_next_obs = np.concatenate(multiframe_state, axis=0)

                    ship_reward = env.get_mf_collector_ship_reward(
                        observation=observation,
                        converted_observation=converted_next_obs,
                        uid=ship_id,
                        done=terminal
                    )

                    next_state_vector = converted_next_obs.flatten()
                    self.ship_agent.remember(state=s, action=a,
                                             reward=ship_reward, new_state=next_state_vector, done=terminal)
                    self.ship_agent.learn(step_num=self.step_number, episode_num=self.episode_number)

            for shipyard_id, val in shipyard_simulated_step_memory.items():
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
                shipyard_reward = env.get_shipyard_count_reward(
                    observation=observation,
                    converted_observation=converted_next_obs,
                    uid=shipyard_id,
                    done=terminal
                )
                next_state_vector = converted_next_obs.flatten()
                next_state_vector: np.ndarray = np.append(next_state_vector, is_occupied_next)

                self.shipyard_agent.remember(
                    state=s,
                    action=a,
                    reward=shipyard_reward,
                    new_state=next_state_vector,
                    done=terminal
                )
                self.shipyard_agent.learn(step_num=self.step_number, episode_num=self.episode_number)

        return [item[0] for item in observation.players]

    def get_single_shipyard_move(self,
                                 shipyard_id,
                                 pos,
                                 step_observation,
                                 raw_observation,
                                 shipyard_temporary_initial_memory
                                 ):
        configuration = self.configuration
        board = Board(raw_observation, raw_configuration=configuration)
        observation = Observation(raw_observation)

        verbose = self.verbose
        done = False

        # Select action
        converted_observation, is_occupied = self.env.wrap_observation_for_shipyard_agent(
            obs=observation, player=observation.player, spos=pos, uid=shipyard_id
        )
        state_vector = converted_observation.flatten()
        state_vector: np.ndarray = np.append(state_vector, is_occupied)

        action = self.shipyard_agent.get_action(
            state_vector, step=self.step_number, game=self.episode_number
        )
        halite_action = self.env.convert_shipyard_action_to_halite_enum(action, shipyard_id, observation)
        self.episode_actions.append(halite_action)

        """
        ============
        Take Action
        ============
        """
        obs_next = self.env.update_observation_for_shipyard(board, shipyard_id, halite_action)

        reward = self.env.get_shipyard_count_reward(
            obs_next,
            self.env.wrap_observation_for_ship_agent(
                obs=obs_next,
                player=obs_next.player,
                spos=pos,  # because shipyards can't move
                uid=shipyard_id
            ),
            uid=shipyard_id,
            done=done
        )

        self.episode_rewards.append(reward)

        """
        ============
        Prepare for Update Model
        ============
        """

        if self.training:
            shipyard_temporary_initial_memory[shipyard_id] = {
                'state': state_vector, 'action': action, 'pos': pos, 'is_occupied': is_occupied
            }

        if verbose and ((self.step_number % 5) == 0):
            print(f"Step {self.step_number}: Action taken {action} for shipyard {shipyard_id}, "
                  f"reward received {reward}")
        # update current observation with the simulated step ahead
        raw_observation = obs_next

        return raw_observation

    def get_moves_for_all_shipyards(
            self,
            step_observation,
            raw_observation,
    ):
        shipyard_simulated_step_memory = {}
        for shipyard_id, (pos) in step_observation.players[step_observation.player][1].items():
            # modifies shipyard_simulated_step_memory object
            raw_observation = self.get_single_shipyard_move(
                shipyard_id=shipyard_id,
                pos=pos,
                step_observation=step_observation,
                raw_observation=raw_observation,
                shipyard_temporary_initial_memory=shipyard_simulated_step_memory
            )
        return raw_observation, shipyard_simulated_step_memory

    def get_moves_for_all_ships(
            self,
            step_observation,
            raw_observation,
    ):
        ship_simulated_step_memory = {}
        for ship_id, (pos, halite) in step_observation.players[step_observation.player][2].items():
            # modifies ship_simulated_step_memory
            raw_observation = self.get_single_ship_move(
                ship_id=ship_id,
                pos=pos,
                step_observation=step_observation,
                raw_observation=raw_observation,
                ship_simulated_step_memory=ship_simulated_step_memory
            )
        return raw_observation, ship_simulated_step_memory

    def get_single_ship_move(
            self,
            ship_id,
            pos,
            step_observation,
            raw_observation,
            ship_simulated_step_memory
    ):
        done = False

        board = Board(
            raw_observation,
            raw_configuration=self.configuration
        )
        observation = Observation(raw_observation)

        """
        ============
        Take Action
        ============
        """
        converted_observation = self.env.wrap_observation_for_ship_agent(
            obs=Observation(board.observation),
            player=board.observation['player'],
            spos=int(pos),
            uid=ship_id
        )
        state_vector = converted_observation.flatten()

        if self.ship_frame_stack_len > 1:
            multiframe_state = self.env.get_multiframe_ship_observation(ship_id)
            converted_obs = np.concatenate(multiframe_state, axis=0)
            state_vector = converted_obs.flatten()
        if len(self.env.get_multiframe_ship_observation(ship_id)) == self.ship_frame_stack_len:
            action = self.ship_agent.get_action(
                state_vector, step=self.step_number, game=self.episode_number
            )
        else:
            action = np.random.randint(0, 6)

        self.episode_actions.append(action)

        halite_action = self.env.convert_ship_action_to_halite_enum(action, observation)

        if halite_action and halite_action.name == halite_action.CONVERT.name and \
                observation.players[observation.player][0] < 500:
            # tried to convert without enough halite
            halite_action = None
            action = 5

        if halite_action:
            self.actions_for_step[ship_id] = halite_action.name

        # Take action
        try:
            obs_next: Observation = self.env.update_observation_for_ship(board, ship_id, halite_action)
        except KeyError as e:
            print('Actions taken')
            print(self.actions_for_step)
            print('Initial board and observation')
            print(step_observation.players[step_observation.player])
            raise e

        # the ship may no longer exist...
        # ie it collided with an enemy ship or converted to a shipyard, we need to use the previous
        # for now we will use the new position IF it exists, otherwise just use the old one
        # next_pos = obs_next.players[observation.player][2].get(ship_id, (None, None))[0]

        """
        ============
        Prepare for Model Update
        ============
        """

        # Update model
        if self.training:
            ship_simulated_step_memory[ship_id] = {'state': state_vector, 'action': action, 'pos': pos}
        action_string = halite_action.name if halite_action else 'None'

        if self.verbose and ((self.step_number % 5) == 0):
            print(f"Step {self.step_number}: Action taken {action} | {action_string} for ship {ship_id}, "
                  f"reward received N/A | Player state {obs_next.players[observation.player]}")
        # update current observation with the simulated step ahead
        raw_observation = obs_next

        return raw_observation
