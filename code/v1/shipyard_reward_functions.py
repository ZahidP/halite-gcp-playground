from kaggle_environments.envs.halite.helpers import ShipyardAction, Observation
import numpy as np


def win_loss_reward(observation):
    player_halite = observation.players[observation.player][0]
    opponent_halites = [item[0] for item in observation.players[observation.player:]]
    best_opponent_halite = sorted(opponent_halites, reverse=True)[0]

    if player_halite > best_opponent_halite:
        return 10_000
    else:
        return -10_000


def convert_state_to_shipyard_ship_count_reward(observation, converted_observation, uid=None, done=False):
    if done:
        return win_loss_reward(observation)

    ship_state: np.ndarray = converted_observation[:, :, 1]
    friendly_ship_mask = ship_state[ship_state > 0]
    enemy_ship_mask = ship_state[ship_state < 0]
    friendly_ships = sum(friendly_ship_mask)
    enemy_ships = sum(enemy_ship_mask)
    # 4 ships in a 5x5 area seems like overkill
    if friendly_ships > 3:
        return -1
    return friendly_ships - enemy_ships


def convert_state_to_reward(observation: Observation, converted_observation, uid=None, done=False) -> float:
    if done:
        return win_loss_reward(observation)

    return -1


def convert_state_to_halite_ratio_reward(observation: Observation, converted_observation, uid=None,
                                         done=False) -> float:
    if done:
        return win_loss_reward(observation)

    ships = len(observation.players[observation.player][2])

    area_halite = 5

    if ships / observation.halite:
        pass

    return -1
