from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Observation

from ship_state_wrapper import ShipStateWrapper
from shipyard_state_wrapper import ShipYardStateWrapper

from kaggle_environments import make


class HaliteEnv:
    def __init__(self, opponents,
                 shipyard_state_wrapper: ShipYardStateWrapper,
                 ship_state_wrapper: ShipStateWrapper,
                 replay_save_dir='',
                 agent_count=1,
                 radius=5,
                 trainer=None,
                 configuration=None,
                 **kwargs):
        self.shipyard_state_wrapper = shipyard_state_wrapper
        self.ship_state_wrapper = ship_state_wrapper

        self.environment = make('halite', configuration)
        self.trainer = trainer if trainer else self.environment.train([None, *opponents])
        self.environment.reset(agent_count)
        self.replay_save_dir = replay_save_dir
        self.state = self.environment.state[0]
        self.observation = None
        self.radius = radius
        self.episode = 0
        self.step_number = 0

    def update_observation_for_ship(self, board: Board, uid, action):
        """Simulate environment step forward and update observation
        https://www.kaggle.com/sam/halite-sdk-overview#Simulating-Actions-(Lookahead)
        """
        ship = board.ships[uid]
        ship.next_action = action
        ret_val = board.next()
        return Observation(ret_val.observation)

    def update_observation_for_shipyard(self, board: Board, uid, action):
        """Simulate environment step forward and update observation
        https://www.kaggle.com/sam/halite-sdk-overview#Simulating-Actions-(Lookahead)
        """
        ship = board.shipyards[uid]
        ship.next_action = action
        ret_val = board.next()
        return Observation(ret_val.observation)

    def reset(self):
        """Reset trainer environment"""
        self.observation = self.trainer.reset()
        board = Board(self.observation, self.environment.configuration)
        return self.observation, board

    def step(self, actions):
        """Step forward in actual environment"""
        self.observation, reward, terminal, info = self.trainer.step(actions)
        self.observation = Observation(self.observation)
        terminal = terminal if terminal else 0
        return self.observation, reward, terminal

    def wrap_observation_for_ship_agent(self, obs, player, uid, spos):
        return self.ship_state_wrapper.get_basic_single_frame_complete_observation(
            obs, player, spos, uid
        )

    def wrap_observation_for_shipyard_agent(self, obs, player, uid, spos):
        return self.shipyard_state_wrapper.get_basic_single_frame_complete_observation(
            obs, player, spos, uid
        )

    def convert_ship_action_to_halite_enum(self, action, obs):
        action_string = self.ship_state_wrapper.convert_action_to_enum(action=action, obs=obs)
        return action_string

    def convert_shipyard_action_to_halite_enum(self, action, uid, obs):
        return self.shipyard_state_wrapper.convert_action_to_enum(action=action, uid=uid, obs=obs)

    def get_ship_reward(self, observation, converted_observation) -> float:
        return self.ship_state_wrapper.convert_state_to_reward(observation, converted_observation)

    def get_shipyard_reward(self, observation, converted_observation) -> float:
        return self.shipyard_state_wrapper.convert_state_to_reward(observation, converted_observation)

    def get_collector_ship_reward(self, observation, converted_observation, uid) -> float:
        return self.ship_state_wrapper.convert_state_to_collector_ship_reward(
            observation, converted_observation, uid
        )

    def get_destroyer_ship_reward(self, observation, converted_observation) -> float:
        return self.ship_state_wrapper.convert_state_to_destroyer_ship_reward(
            observation, converted_observation
        )

    def get_shipyard_count_reward(self, observation, converted_observation) -> float:
        return self.shipyard_state_wrapper.convert_state_to_shipyard_ship_count_reward(
            observation, converted_observation
        )
