from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Observation
from halite_env import HaliteEnv
from ship_state_wrapper import ShipStateWrapper
from shipyard_state_wrapper import ShipYardStateWrapper
from agent import Agent
from game_runner_v2 import GameRunner
from halite_agent import HaliteAgent

environment = make('halite', {}, debug=True)

halite_env = HaliteEnv(
    opponents=3,
    ship_state_wrapper={},
    shipyard_state_wrapper={},
    radius=4,
    environment=None,
    trainer=None
)


halite_agent = HaliteAgent(
    configuration=environment.configuration,
    halite_env=halite_env,
    ship_agent=ship_agent,
    shipyard_agent=shipyard_agent,
    training=False,
    verbose=False,
    ship_frame_stack_len=2
)

def halite_run_agent(observation, configuration):

    raw_observation = observation
    step_observation = Observation(observation)

    raw_observation, shipyard_simulated_step_memory = halite_agent.get_moves_for_all_shipyards(
        raw_observation=raw_observation,
        step_observation=step_observation,
        episode_number=self.episode_number,
        step_number=self.step_number
    )

    raw_observation, ship_simulated_step_memory = halite_agent.get_moves_for_all_ships(
        raw_observation=raw_observation,
        step_observation=step_observation,
        episode_number=self.episode_number,
        step_number=self.step_number
    )

    actions_for_step = halite_agent.actions_for_step
    return actions_for_step