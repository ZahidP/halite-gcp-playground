from copy import deepcopy
import numpy as np
from kaggle_environments.envs.halite.helpers import ShipyardAction, Observation

MAP_SIZE = 21


class ShipYardStateWrapper:

    def __init__(self, radius=4, max_frames=2, map_size=21, spawn_cost=500):

        # we might want to keep a previous frame
        # if we want to calculate a negative reward of a ship being destroyed
        # this may be necessary
        self.MAP_SIZE = map_size
        self.frame_history = []
        self.radius = radius
        self.spawn_cost = spawn_cost
        self.max_frames = max_frames
        self.cached_ships_map, self.cached_shipyards_map = None, None
        self.state_size = ((2 * radius + 1) ** 2) * 3 + 1

    def set_map_size(self, map_size):
        self.MAP_SIZE = map_size

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
                    entities_map[row, col] = -1
                else:
                    entities_map[row, col] = 1
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

    def get_is_shipyard_occupied(self, obs, sy_pos=None, uid=None):
        phalite, shipyards, ships = obs.players[obs.player]
        if sy_pos:
            occupied = sy_pos in [x[0] for x in ships.values()]
        else:
            occupied = shipyards[uid] in [x[0] for x in ships.values()]
        occupied = 1 if occupied else 0
        return np.array([occupied])

    def convert_action_to_enum(self, uid, obs, action):
        phalite, shipyards, ships = obs.players[obs.player]
        occupied = shipyards[uid] in [x[0] for x in ships.values()]
        if action == 1 and phalite >= self.spawn_cost and not occupied:
            return ShipyardAction.SPAWN
        return None

    def get_basic_single_frame_complete_observation(self, obs, player, sy_pos, uid=None):
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
        # it's kind of dumb to keep occupied in here, we will end up with a big block
        # of either ones or zeros
        is_occupied = self.get_is_shipyard_occupied(obs, sy_pos=sy_pos, uid=uid)

        self.cached_ships_map = ships_map
        self.cached_shipyards_map = shipyards_map

        state_map = np.stack([halite_map, ships_map, shipyards_map], axis=2)
        # this allows the wrap-around functionality
        state_map = np.tile(state_map, (3, 3, 1))

        # we add this MAP_SIZE after the division because of the tiling above
        y = sy_pos // self.MAP_SIZE + self.MAP_SIZE
        x = sy_pos % self.MAP_SIZE + self.MAP_SIZE
        r = self.radius

        return state_map[y - r:y + r + 1, x - r:x + r + 1], is_occupied

    def get_basic_multiframe_observation(self, player, spos, uid):

        combined = []

        for frame in self.frame_history:
            frame_result = self.get_basic_single_frame_complete_observation(obs=frame, player=player, spos=spos)
            combined.append(frame_result)

        return np.stack(combined, axis=2)
