from copy import deepcopy
import numpy as np

MAP_SIZE = 21


class ShipStateWrapper:

    def __init__(self, obs, player, radius=4, max_frames=2):

        # we might want to keep a previous frame
        # if we want to calculate a negative reward of a ship being destroyed
        # this may be necessary
        self.frame_history = []
        self.radius = 4
        self.obs = deepcopy(obs)
        self.player = player
        self.phalite, self.shipyards, self.ships = obs.players[obs.player]
        self.max_frames = max_frames

    def update_state(self, obs):
        self.obs = deepcopy(obs)
        self.phalite, self.shipyards, self.ships = obs.players[obs.player]

        self.frame_history.append(obs)

        if len(self.frame_history) > self.max_frames:
            self.frame_history.pop(0)

    def get_frame_ship_state(self, obs, current_player, uid):
        """
        Returns 4 ship states with locations of friendly and enemy ships.
        The attempt here is to use separate "features" for different ship categories
        :param uid:
        :return:
        """
        phalite, shipyards, ships = obs.players[current_player]

        if uid not in ships:
            # this ship no longer exists
            target_halite = 0
        else:
            target = self.ships[uid]
            target_pos, target_halite = target

        #
        # ship_mode = 'SEARCH'
        #
        # if target_halite < 100:
        #     ship_mode = 'ATTACK'
        # if target_halite > 500:
        #     ship_mode = 'DEPOSIT'

        max_enemy_value = 0

        # === Enemy Ships: To Attack/Ignore ===
        # >500 is a heuristic, consider updating
        enemy_entities_map_attack = np.zeros((MAP_SIZE, MAP_SIZE))
        for player, (_, sy, ship) in enumerate(obs.players):
            if player != current_player:
                for ship in ship.values():
                    ship_pos, halite = ship[0], ship[1]
                    if halite > target_halite or (halite > 500):
                        enemy_entities_map_attack[ship_pos // MAP_SIZE,
                                                  ship_pos % MAP_SIZE] = 1

        # === Enemy Ships: To Evade ===
        # <100 is a heuristic, consider removing
        enemy_entities_map_evade = np.zeros((MAP_SIZE, MAP_SIZE))
        for player, (_, sy, ship) in enumerate(obs.players):
            if player != current_player:
                for ship in ship.values():
                    ship_pos, halite = ship[0], ship[1]
                    if halite > target_halite:
                        enemy_entities_map_evade[ship_pos // MAP_SIZE,
                                                 ship_pos % MAP_SIZE] = 1

        friendly_ships = ships

        # === Friendly Ships: Bodyguards
        # Idea here is that these agents will "know" to attack enemy ships
        # <100 is a heuristic, consider removing
        bodyguards_count = 0
        bodyguards_entities_map = np.zeros((MAP_SIZE, MAP_SIZE))
        for ship in friendly_ships.values():
            ship_pos, halite = ship[0], ship[1]
            if halite > max_enemy_value or (halite < 100):
                bodyguards_entities_map[ship_pos // MAP_SIZE,
                                        ship_pos % MAP_SIZE] = 1

        # === Friendly Ships: Need Assistance
        # Not sure if this one will be helpful or add to much complexity
        # >500 is a heuristic, consider updating
        need_assistance_entities_map = np.zeros((MAP_SIZE, MAP_SIZE))
        for ship in friendly_ships.values():
            ship_pos, halite = ship[0], ship[1]
            if halite < max_enemy_value or (halite > 500):
                need_assistance_entities_map[ship_pos // MAP_SIZE,
                                             ship_pos % MAP_SIZE] = 1

        return bodyguards_entities_map, need_assistance_entities_map, enemy_entities_map_attack, enemy_entities_map_evade

    def get_frame_shipyard_state(self, obs, current_player):
        """
        This returns just one np.array with +1 for friendly shipyards and -1 for
        enemy shipyards.

        :return:
        """

        entities_map = np.zeros((MAP_SIZE, MAP_SIZE))

        for player, (_, shipyard, s) in enumerate(obs.players):
            shipyard_pos, halite = shipyard
            if player != current_player:
                # === Enemy Shipyards ===
                entities_map[shipyard_pos // MAP_SIZE, shipyard_pos % MAP_SIZE] = -1
            else:
                # === Friendly Shipyards
                entities_map[shipyard_pos // MAP_SIZE, shipyard_pos % MAP_SIZE] = 1

        return entities_map

    def get_single_frame_complete_observation(self, obs, player, spos, uid):
        """
        Here we derive the converted state w.r.t a position and not a uid
        :param spos:
        :param uid:
        :return:
        """
        halite_map = np.reshape(self.obs.halite, (MAP_SIZE, MAP_SIZE))

        obs = deepcopy(obs)

        ships_map = self.get_frame_ship_state(obs=obs, current_player=player, uid=uid)
        shipyard_map = self.get_frame_shipyard_state(obs=obs, current_player=player)

        state_map = np.stack([halite_map, ships_map, shipyard_map], axis=2)
        # this allows the wrap-around functionality
        state_map = np.tile(state_map, (3, 3, 1))

        y = spos // MAP_SIZE + MAP_SIZE
        x = spos % MAP_SIZE + MAP_SIZE
        r = self.radius
        return state_map[y - r:y + r + 1, x - r:x + r + 1]

    def get_multiframe_observation(self, player, spos, uid):

        combined = []

        for frame in self.frame_history:
            frame_result = self.get_single_frame_complete_observation(obs=frame, player=player, spos=spos, uid=uid)
            combined.append(frame_result)

        return np.stack(combined, axis=2)

