from copy import deepcopy
import numpy as np
from kaggle_environments.envs.halite.helpers import ShipyardAction, Observation


# assuming just a single state
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