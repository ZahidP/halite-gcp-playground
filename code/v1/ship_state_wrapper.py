from copy import deepcopy
from collections import deque, defaultdict
from kaggle_environments.envs.halite.helpers import ShipAction, Observation

import numpy as np

MAP_SIZE = 21


class ShipStateWrapper:

    def __init__(self, radius=4, max_frames=2, map_size=21):

        # we might want to keep a previous frame
        # if we want to calculate a negative reward of a ship being destroyed
        # this may be necessary
        self.MAP_SIZE = map_size
        self.frame_history = defaultdict(lambda: deque(maxlen=max_frames))
        self.radius = radius
        self.obs = None
        self.max_frames = max_frames
        self.cached_ships_map, self.cached_shipyards_map = None, None
        self.state_size = (((2 * radius + 1) ** 2) * 3 + 3) * max_frames

    def set_map_size(self, map_size):
        self.MAP_SIZE = map_size

    def reset(self):
        self.frame_history = {}

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

    def get_basic_single_frame_map_observation(self, obs, player: int, spos: int, uid=None) -> np.ndarray:
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

        result = state_map[y - r:y + r + 1, x - r:x + r + 1]

        return result

    def get_basic_single_frame_complete_observation(
            self, obs: Observation, player: int, spos: int, remaining, turn, uid=None, map_size=None) -> np.ndarray:
        """
        Here we derive the converted state w.r.t a position and not a uid
        :param spos:
        :param uid:
        :return:
        """

        if map_size:
            self.MAP_SIZE = map_size

        result = self.get_basic_single_frame_map_observation(obs, player, spos, uid)

        player_halite = obs.players[obs.player][0]
        remaining_ship_moves = remaining
        turn = turn

        result = result.flatten()

        result = np.concatenate([result, [player_halite], [remaining_ship_moves], [turn]])

        # assuming this is pass by reference
        self.frame_history[uid].appendleft(result)
        return result

    def get_basic_state_history(self, uid):

        combined = []
        uid_history = self.frame_history.get(uid, deque(maxlen=self.max_frames))
        for frame in uid_history:
            combined.append(frame)
        return combined
