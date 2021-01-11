import numpy as np
from kaggle_environments.envs.halite.helpers import Board, Observation, ShipAction
from shipyard_state_wrapper import ShipYardStateWrapper
from ship_state_wrapper import ShipStateWrapper
from halite_env import HaliteEnv
from typing import Dict, Any


class HaliteAgent:

    def __init__(self,
                 configuration,
                 ship_agent,
                 shipyard_agent,
                 halite_env: HaliteEnv,
                 verbose=False,
                 training=False,
                 ship_frame_stack_len=1,
                 shipyard_frame_stack_len=1
                 ):
        self.configuration = configuration
        self.ship_agent = ship_agent
        self.shipyard_agent = shipyard_agent
        self.verbose = verbose
        self.training = training
        self.ship_state_wrapper = ShipStateWrapper(
            radius=4,
            max_frames=2,
            map_size=configuration['size']
        )
        self.shipyard_state_wrapper = ShipYardStateWrapper(
            radius=4,
            max_frames=1,
            map_size=configuration['size']
        )
        self.env = halite_env
        self.ship_frame_stack_len = ship_frame_stack_len
        self.shipyard_frame_stack_len = shipyard_frame_stack_len
        self.actions_for_step = {}
        self.remaining = {}

    def __call__(
            self, observation: Dict[str, Any], configuration: Dict[str, Any]
    ) -> Dict[Any, str]:
        raw_observation = observation
        step_observation = Observation(observation)

        raw_observation, shipyard_simulated_step_memory = self.get_moves_for_all_shipyards(
            raw_observation=raw_observation,
            step_observation=step_observation,
            episode_number=0,
            step_number=0
        )

        self.get_moves_for_all_ships(
            raw_observation=raw_observation,
            step_observation=step_observation,
            episode_number=0,
            step_number=0
        )

        actions_for_step = self.actions_for_step
        return actions_for_step

    def reset_step_actions(self):
        self.actions_for_step = {}

    def learn(
            self,
            observation,
            game_reward,
            terminal,
            ship_simulated_step_memory,
            shipyard_simulated_step_memory,
            step_number,
            episode_number
    ):
        env = self.env

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
        remaining = len(ship_simulated_step_memory)

        for ship_id, val in ship_simulated_step_memory.items():
            s = val['state']
            a = val['action']
            pos = val['pos']

            if step_number >= self.ship_frame_stack_len:

                if self.ship_frame_stack_len > 1:
                    multiframe_state = self.env.get_multiframe_ship_observation(ship_id)
                    converted_obs = np.concatenate(multiframe_state, axis=0)
                    s = converted_obs.flatten()

                converted_next_obs = env.wrap_observation_for_ship_agent(
                    obs=Observation(observation),
                    player=observation['player'],
                    remaining=remaining,
                    turn=step_number,
                    spos=int(pos),
                    uid=ship_id
                )

                remaining -= 1

                if self.ship_frame_stack_len > 1:
                    multiframe_state = self.env.get_multiframe_ship_observation(ship_id)
                    converted_next_obs = np.concatenate(multiframe_state, axis=0)

                ship_reward = env.get_ship_reward(
                    observation=observation,
                    converted_observation=converted_next_obs,
                    uid=ship_id,
                    done=terminal
                )

                if len(converted_next_obs) >= self.ship_frame_stack_len:
                    next_state_vector = converted_next_obs.flatten()
                    try:
                        self.ship_agent.remember(state=s, action=a,
                                                 reward=ship_reward, new_state=next_state_vector, done=terminal)
                        self.ship_agent.learn(step_num=step_number, episode_num=episode_number)
                    except Exception as e:
                        print('shapes')
                        print(s.shape)
                        print(next_state_vector.shape)
                        raise e

        for shipyard_id, val in shipyard_simulated_step_memory.items():
            s = val['state']
            a = val['action']
            pos = val['pos']
            converted_next_obs = env.wrap_observation_for_shipyard_agent(
                obs=Observation(observation),
                player=observation['player'],
                spos=int(pos),
                uid=shipyard_id
            )
            shipyard_reward = env.get_shipyard_reward(
                observation=observation,
                converted_observation=converted_next_obs,
                uid=shipyard_id,
                done=terminal
            )
            next_state_vector = converted_next_obs.flatten()

            self.shipyard_agent.remember(
                state=s,
                action=a,
                reward=shipyard_reward,
                new_state=next_state_vector,
                done=terminal
            )
            self.shipyard_agent.learn(step_num=step_number, episode_num=episode_number)

        self.reset_step_actions()
        return terminal, [item[0] for item in observation.players]

    def get_single_shipyard_move(self,
                                 shipyard_id,
                                 pos,
                                 step_observation,
                                 raw_observation,
                                 shipyard_temporary_initial_memory,
                                 step_number=0,
                                 episode_number=0,
                                 ):
        configuration = self.configuration
        board = Board(raw_observation, raw_configuration=configuration)
        observation = Observation(raw_observation)

        verbose = self.verbose
        done = False

        # Select action
        converted_observation = self.shipyard_state_wrapper.get_basic_single_frame_complete_observation(
            obs=observation, player=observation.player, sy_pos=pos, uid=shipyard_id
        )
        state_vector = converted_observation

        player_state = step_observation.players[step_observation.player]

        if len(player_state[2]) == 0 and player_state[0] > 500:
            action = 1
        else:
            action = self.shipyard_agent.get_action(
                state_vector, step=step_number, game=episode_number
            )
        halite_action = self.shipyard_state_wrapper.convert_action_to_enum(shipyard_id, observation, action)

        if halite_action:
            self.actions_for_step[shipyard_id] = halite_action.name

        """
        ============
        Take Action
        ============
        """
        obs_next = self.env.update_observation_for_shipyard(board, shipyard_id, halite_action)

        reward = self.env.get_shipyard_reward(
            obs_next,
            self.env.wrap_observation_for_shipyard_agent(
                obs=obs_next,
                player=obs_next.player,
                spos=pos,  # because shipyards can't move
                uid=shipyard_id
            ),
            uid=shipyard_id,
            done=done
        )

        """
        ============
        Prepare for Update Model
        ============
        """

        is_occupied = state_vector[-2]

        if self.training:
            shipyard_temporary_initial_memory[shipyard_id] = {
                'state': state_vector, 'action': action, 'pos': pos, 'is_occupied': is_occupied
            }

        if verbose and ((step_number % 10) == 0):
            print(f"Step {step_number}: Action taken {action} for shipyard {shipyard_id}, "
                  f"reward received {reward}")
        # update current observation with the simulated step ahead
        raw_observation = obs_next

        return raw_observation

    def get_moves_for_all_shipyards(
            self,
            step_observation,
            raw_observation,
            episode_number,
            step_number
    ):
        shipyard_simulated_step_memory = {}

        for shipyard_id, (pos) in step_observation.players[step_observation.player][1].items():
            # modifies shipyard_simulated_step_memory object
            raw_observation = self.get_single_shipyard_move(
                shipyard_id=shipyard_id,
                pos=pos,
                step_observation=step_observation,
                raw_observation=raw_observation,
                shipyard_temporary_initial_memory=shipyard_simulated_step_memory,
                episode_number=episode_number,
                step_number=step_number
            )
        return raw_observation, shipyard_simulated_step_memory

    def get_moves_for_all_ships(
            self,
            step_observation,
            raw_observation,
            episode_number,
            step_number,
    ):

        ship_simulated_step_memory = {}
        remaining_ships = list(step_observation.players[step_observation.player][2].items())[:]  # make copy

        while remaining_ships:
            remaining = len(remaining_ships)
            ship_id, (pos, halite) = remaining_ships.pop(0)
            # modifies ship_simulated_step_memory
            if ship_id not in raw_observation['players'][raw_observation['player']][2].keys():
                print(f'Ship {ship_id} most likely collided, skipping this portion')
                continue

            raw_observation = self.get_single_ship_move(
                ship_id=ship_id,
                pos=pos,
                step_observation=step_observation,
                raw_observation=raw_observation,
                ship_simulated_step_memory=ship_simulated_step_memory,
                episode_number=episode_number,
                step_number=step_number,
                remaining=remaining
            )
        return raw_observation, ship_simulated_step_memory

    def get_single_ship_move(
            self,
            ship_id,
            pos,
            step_observation,
            raw_observation,
            ship_simulated_step_memory,
            step_number,
            episode_number,
            remaining
    ):
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
        self.env.wrap_observation_for_ship_agent(
            obs=Observation(board.observation),
            player=board.observation['player'],
            remaining=remaining,
            turn=step_number,
            spos=int(pos),
            uid=ship_id
        )

        multiframe_state = self.env.get_multiframe_ship_observation(ship_id)
        converted_obs = np.concatenate(multiframe_state, axis=0)
        state_vector = converted_obs.flatten()
        if not len(observation.players[observation.player][1]) and \
                (observation.players[observation.player][0] >= 1000):
            # convert if possible
            action = 4
        else:
            if len(self.env.get_multiframe_ship_observation(ship_id)) == self.ship_frame_stack_len:
                action = self.ship_agent.get_action(
                    state_vector, step=step_number, game=episode_number
                )
            else:
                action = np.random.randint(0, 6)

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
            print('Ship {} most likely collided with a friendly ship'.format(ship_id))

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
        if self.training and len(multiframe_state) >= self.ship_frame_stack_len:
            ship_simulated_step_memory[ship_id] = {'state': state_vector, 'action': action, 'pos': pos}
        action_string = halite_action.name if halite_action else 'None'

        if self.verbose and ((step_number % 10) == 0):
            print(f"Step {step_number}: Action taken {action} | {action_string} for ship {ship_id}, "
                  f"reward received N/A | Player state {obs_next.players[observation.player]}")
        # update current observation with the simulated step ahead
        raw_observation = obs_next
        return raw_observation
