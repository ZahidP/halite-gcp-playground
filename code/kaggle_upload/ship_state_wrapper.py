from copy import deepcopy
from kaggle_environments.envs.halite.helpers import ShipAction, Observation

import numpy as np

MAP_SIZE = 21


class ShipStateWrapper:

    def __init__(self, radius=4, max_frames=2, map_size=21):

        # we might want to keep a previous frame
        # if we want to calculate a negative reward of a ship being destroyed
        # this may be necessary
        self.MAP_SIZE = map_size
        self.frame_history = []
        self.radius = radius
        self.obs = None
        self.max_frames = max_frames
        self.cached_ships_map, self.cached_shipyards_map = None, None
        self.state_size = ((2 * radius + 1) ** 2) * 3

    def convert_state_to_collector_ship_reward(self, observation: Observation, converted_observation, uid) -> float:
        player_halite = observation.players[observation.player][0]
        player_ships = observation.players[observation.player][2]
        ship_status = player_ships.get(uid)
        if ship_status:
            ship_halite = ship_status[1] * 0.1
            if ship_halite > 600:
                ship_halite = -10
        else:
            # if we converted this -10 is somewhat misleading
            ship_halite = -10
        return player_halite + ship_halite

    def convert_state_to_destroyer_ship_reward(self, observation: Observation, converted_observation) -> float:
        opponent_halites = [item[0] for item in observation.players[observation.player:]]
        best_opponent_halite = sorted(opponent_halites, reverse=True)[0]
        return -best_opponent_halite

    def convert_state_to_reward(self, observation: Observation, converted_observation) -> float:

        player_halite = observation.players[observation.player][0]

        opponent_halites = [item[0] for item in observation.players[observation.player:]]

        best_opponent_halite = sorted(opponent_halites, reverse=True)[0]

        halite_state = converted_observation[:, :, 0]
        ship_state: np.ndarray = converted_observation[:, :, 1]
        shipyard_state: np.ndarray = converted_observation[:, :, 2]

        # basically, my ships minus their ships
        ship_rewards = ship_state.sum()
        # basically, my shipyards minus their shipyards
        shipyard_rewards = shipyard_state.sum()

        """
        This is kind of arbitrary. 
        I think the most important measure is total halite (minus opponent's halite).
        Ship halite is probably somewhat over-weighted, especially because those ships are vulnerable.
        
        One thing I don't want to do is overemphasize a bunch of corner cases in terms of reward functions.
        The point of reinforcement learning is to be able to determine which scenarios result in greater overall 
        discounted reward instead of the programmer having to specify these rewards.
        
        That being said it may be helpful to nudge things in the right direction.
        
        I divide ship rewards by 5 because halite on ships is not as valuable as halite stored.
        I multiply shipyard rewards because having a shipyard is a good thing and they are currently 
        in the state as +/- 1, which is inconsequential, it would take a long time for the agent to potentially learn
        that converting to a shipyard is a good thing because of the associated cost.
        
        """
        try:
            # extra_rewards = (ship_rewards ** (1/3)) + shipyard_rewards
            return (player_halite - best_opponent_halite)
        except RuntimeWarning as w:
            print(w)
            print(f'Halite differential {((player_halite - best_opponent_halite) * 2)}')
            print(f'Ship Rewards: {ship_rewards}')
            return ((player_halite - best_opponent_halite) * 2)

    def convert_action_to_enum(self, obs: Observation, action=0) -> ShipAction:
        action_strings = {
            0: ShipAction.NORTH,
            1: ShipAction.SOUTH,
            2: ShipAction.EAST,
            3: ShipAction.WEST,
            4: ShipAction.CONVERT,
            5: None
        }

        action_string = action_strings[action]

        if (obs.players[obs.player][0] < 500) and \
                (action_string) and \
                (action_string.name == ShipAction.CONVERT.name):
            return action_strings[np.random.randint(0, 4)]
        else:
            return action_string

    def update_state(self, obs):
        self.obs = deepcopy(obs)
        self.phalite, self.shipyards, self.ships = obs.players[obs.player]

        self.frame_history.append(obs)

        if len(self.frame_history) > self.max_frames:
            self.frame_history.pop(0)

    def get_frame_ship_state_basic(self, obs, current_player):
        """
        This returns just one np.array with +{halite} for friendly ships and -{halite} for
        enemy ships.


        :return:
        """

        entities_map = np.zeros((self.MAP_SIZE, self.MAP_SIZE))

        for player, (_, sy, ships) in enumerate(obs.players):
            for ship in ships.values():
                ship_pos, halite = ship[0], ship[1]
                row = ship_pos // self.MAP_SIZE
                col = ship_pos % self.MAP_SIZE
                if player != current_player:
                    entities_map[row, col] = -halite if halite > 0 else -1
                else:
                    entities_map[row, col] = halite if halite > 0 else 1
        return entities_map

    def get_frame_shipyard_state_basic(self, obs, current_player):
        """
        This returns just one np.array with +1 for friendly shipyards and -1 for
        enemy shipyards.

        :return:
        """

        entities_map = np.zeros((self.MAP_SIZE, self.MAP_SIZE))

        for player, (_, shipyards, s) in enumerate(obs.players):
            for shipyard_pos in shipyards.values():
                if player != current_player:
                    # === Enemy Shipyards ===
                    entities_map[shipyard_pos // self.MAP_SIZE, shipyard_pos % MAP_SIZE] = -1
                else:
                    # === Friendly Shipyards
                    entities_map[shipyard_pos // self.MAP_SIZE, shipyard_pos % self.MAP_SIZE] = 1

        return entities_map

    def get_basic_single_frame_complete_observation(self, obs, player: int, spos: int, uid=None) -> np.ndarray:
        """
        Here we derive the converted state w.r.t a position and not a uid
        :param spos:
        :param uid:
        :return:
        """
        halite_map = np.reshape(obs.halite, (self.MAP_SIZE, self.MAP_SIZE))

        obs = deepcopy(obs)

        ships_map = self.get_frame_ship_state_basic(obs=obs, current_player=player)
        shipyards_map = self.get_frame_shipyard_state_basic(obs=obs, current_player=player)

        self.cached_ships_map = ships_map
        self.cached_shipyards_map = shipyards_map

        state_map = np.stack([halite_map, ships_map, shipyards_map], axis=2)
        # this allows the wrap-around functionality
        state_map = np.tile(state_map, (3, 3, 1))

        # we add this MAP_SIZE after the division because of the tiling above
        y = spos // self.MAP_SIZE + self.MAP_SIZE
        x = spos % self.MAP_SIZE + self.MAP_SIZE
        r = self.radius

        return state_map[y - r:y + r + 1, x - r:x + r + 1]

    def get_basic_multiframe_observation(self, player, spos, uid):

        combined = []

        for frame in self.frame_history:
            frame_result = self.get_basic_single_frame_complete_observation(obs=frame, player=player, spos=spos)
            combined.append(frame_result)

        return np.stack(combined, axis=2)
