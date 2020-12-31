import numpy as np
from kaggle_environments.envs.halite.helpers import Board, Observation
from halite_agent import HaliteAgent
from halite_env import HaliteEnv


class GameRunner:

    def __init__(self,
                 env,
                 shipyard_agent,
                 ship_agent,
                 training,
                 configuration,
                 max_steps=100,
                 verbose=True,
                 ship_frame_stack_len=1,
                 shipyard_frame_stack_len=1,
                 handicap=0
                 ):
        # code smell...too many properties here
        self.actions = {}
        self.env: HaliteEnv = env
        self.shipyard_agent = shipyard_agent
        self.ship_agent = ship_agent
        self.episode_number = 0
        self.step_number = 0
        self.episode_actions = []
        self.episode_rewards = []
        self.training = training
        self.configuration = configuration
        self.verbose = verbose
        self.actions_for_step = {}
        self.max_steps = max_steps
        self.episode_score = 0
        self.shipyard_step_memory = {}
        self.ship_step_memory = {}
        self.ship_frame_stack_len = ship_frame_stack_len
        self.shipyard_frame_stack_len = shipyard_frame_stack_len
        self.handicap = handicap
        self.halite_agent = HaliteAgent(
            configuration=configuration,
            halite_env=env,
            ship_agent=ship_agent,
            shipyard_agent=shipyard_agent,
            training=True,
            ship_frame_stack_len=ship_frame_stack_len
        )

    def play_episode(self, max_steps):
        raw_observation: dict = self.env.reset()[0].__dict__
        raw_observation[
            'players'
        ][raw_observation['player']][0] = raw_observation['players'][raw_observation['player']][0] - self.handicap

        episode_scores = []

        if self.episode_number % 5 == 0:
            print("Episode {}".format(self.episode_number))
            print(raw_observation['players'])

        for step in range(max_steps):
            self.step_number = step
            step_observation = Observation(raw_observation)
            raw_observation, done, step_scores = self.play_step(
                raw_observation=raw_observation,
                step_observation=step_observation
            )
            episode_scores.append(step_scores)
            if done:
                return episode_scores
        self.episode_number = 1 + self.episode_number
        return episode_scores

    def play_step(self, raw_observation, step_observation):
        self.actions_for_step = {}
        self.shipyard_step_memory = {}
        self.ship_step_memory = {}

        raw_observation, shipyard_simulated_step_memory = self.halite_agent.get_moves_for_all_shipyards(
            raw_observation=raw_observation,
            step_observation=step_observation,
            episode_number=self.episode_number,
            step_number=self.step_number
        )

        raw_observation, ship_simulated_step_memory = self.halite_agent.get_moves_for_all_ships(
            raw_observation=raw_observation,
            step_observation=step_observation,
            episode_number=self.episode_number,
            step_number=self.step_number
        )

        env = self.env
        for id_, action in self.halite_agent.actions_for_step.items():
            self.actions_for_step[id_] = action

        # updates the env.observation
        step_results = env.step(actions=self.actions_for_step)

        observation, game_reward, terminal = step_results

        if self.training:
            if self.step_number > self.ship_frame_stack_len:
                self.halite_agent.learn(
                    observation=observation,
                    game_reward=game_reward,
                    terminal=terminal,
                    ship_simulated_step_memory=ship_simulated_step_memory,
                    shipyard_simulated_step_memory=shipyard_simulated_step_memory,
                    episode_number=self.episode_number,
                    step_number=self.step_number
                )

        return observation, terminal, [item[0] for item in observation.players]
