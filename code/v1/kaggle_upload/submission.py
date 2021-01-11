from kaggle_environments.envs.halite.helpers import *
from random import choice

from agent import Agent
from ship_state_wrapper import ShipStateWrapper
from shipyard_state_wrapper import ShipYardStateWrapper

def agent(obs,config):

    board = Board(obs,config)
    me = board.current_player

    ship_wrapper = ShipStateWrapper(radius=4, max_frames=2, map_size=int(board.configuration['size']))
    shipyard_wrapper = ShipStateWrapper(radius=4, max_frames=2, map_size=int(board.configuration['size']))

    ship_agent = Agent(
        alpha=0.99,
        gamma=0.2,
        batch_size=32,
        epsilon=0.99,
        epsilon_end=.1,
        epsilon_dec=0.99, input_dims=ship_wrapper.state_size,
        n_actions=6,
        fname='ship_agent.h5'
    )

    ship_agent = Agent(
        alpha=0.99,
        gamma=0.2,
        batch_size=32,
        epsilon=0.99,
        epsilon_end=.1,
        epsilon_dec=0.99, input_dims=shipyard_wrapper.state_size,
        n_actions=2,
        fname='ship_agent.h5'
    )

    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = choice([ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,None])

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None

    return me.next_actions
