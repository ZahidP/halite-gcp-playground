from copy import deepcopy
import numpy as np
from kaggle_environments.envs.halite.helpers import ShipyardAction, Observation


def win_loss_reward(observation):
    player_halite = observation.players[observation.player][0]
    opponent_halites = [item[0] for item in observation.players[observation.player:]]
    best_opponent_halite = sorted(opponent_halites, reverse=True)[0]

    if player_halite > best_opponent_halite:
        return 1_000
    else:
        return -1_000


# assuming just a single state
def convert_state_to_collector_ship_reward(
        observation: Observation,
        converted_observation,
        uid=None,
        done=False
) -> float:
    """
    The purpose of this function is to incentivize the agent to collect and deposit halite.

    We will make use of two state frames.
    """

    print('shape: {}'.format(converted_observation.shape))
    print(converted_observation)

    if done:
        return win_loss_reward(observation)

    player_halite = observation.players[observation.player][0]
    # center ship is current ship
    ship_state = converted_observation[:, :, 1]
    ship_halite = ship_state[len(converted_observation) // 2 + 1]

    if ship_halite:
        ship_halite = ship_halite * 0.1
        if ship_halite > 600:
            ship_halite = -10
    else:
        # if we converted this -10 is somewhat misleading
        ship_halite = -10
    return player_halite + ship_halite


def convert_state_to_destroyer_ship_reward(
        observation: Observation,
        converted_observation,
        uid=None,
        done=False
) -> float:
    opponent_halites = [item[0] for item in observation.players[observation.player:]]
    best_opponent_halite = sorted(opponent_halites, reverse=True)[0]
    return -best_opponent_halite


def convert_state_to_reward(observation: Observation, converted_observation, uid=None, done=False) -> float:
    player_halite = observation.players[observation.player][0]

    opponent_halites = [item[0] for item in observation.players[observation.player:]]

    best_opponent_halite = sorted(opponent_halites, reverse=True)[0]

    if done:
        if player_halite > best_opponent_halite:
            return 20_000
        else:
            return -20_000

    return (player_halite - best_opponent_halite) / 2


"""
================================
================================
==== Multi Frame Rewards
================================
================================
"""


def multi_frame_win_loss_ship_reward(
        observation: Observation,
        converted_observation,
        uid=None,
        done=False
) -> float:
    """
    The purpose of this function is to incentivize the agent to collect and deposit halite.

    We will make use of two state frames.
    """

    if done:
        return win_loss_reward(observation)

    return -1


def convert_state_to_collector_reward(
        observation: Observation,
        converted_observation,
        uid=None,
        done=False
) -> float:
    """
    The purpose of this function is to incentivize the agent to collect and deposit halite.

    We will make use of two state frames.
    """

    if done:
        return win_loss_reward(observation)

    current_frame = converted_observation[len(converted_observation) // 2:]
    previous_frame = converted_observation[:len(converted_observation) // 2]

    current_player_halite = current_frame[-3]
    previous_player_halite = previous_frame[-3]
    # center ship is current ship

    current_frame_map_items = current_frame[:-3]
    previous_frame_map_items = previous_frame[:-3]

    map_state_divider = len(current_frame_map_items) // 3
    current_frame_ship_items = current_frame_map_items[map_state_divider: map_state_divider * 2]
    previous_frame_ship_items = previous_frame_map_items[map_state_divider: map_state_divider * 2]

    current_ship_halite = current_frame_ship_items[len(current_frame_ship_items) // 2]
    prev_ship_halite = previous_frame_ship_items[len(previous_frame_ship_items) // 2]

    reward = -20

    accumulated_ship_halite = (current_ship_halite - prev_ship_halite)

    if accumulated_ship_halite > 600:
        accumulated_ship_halite -= (current_ship_halite - 600)

    reward = reward + accumulated_ship_halite * 0.25 + (current_player_halite - previous_player_halite)

    return reward


def convert_state_to_total_halite_reward(
        observation: Observation,
        converted_observation,
        uid=None,
        done=False
) -> float:
    if done:
        return win_loss_reward(observation) * 10

    current_frame = converted_observation[len(converted_observation) // 2:]
    previous_frame = converted_observation[:len(converted_observation) // 2]

    current_player_halite = current_frame[-3]
    previous_player_halite = previous_frame[-3]

    return current_player_halite


def convert_state_to_new_halite_reward(
        observation: Observation,
        converted_observation,
        uid=None,
        done=False
) -> float:
    if done:
        return win_loss_reward(observation) * 10

    current_frame = converted_observation[len(converted_observation) // 2:]
    previous_frame = converted_observation[:len(converted_observation) // 2]

    current_player_halite = current_frame[-3]
    previous_player_halite = previous_frame[-3]

    return current_player_halite - previous_player_halite
