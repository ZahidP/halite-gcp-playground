from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Observation
from unittest import TestCase

from ship_state_wrapper import ShipStateWrapper

test_state = {'action':
                  {'5-1': 'EAST', '4-1': 'SPAWN'},
              'reward': 3042,
              'info': {},
              'observation': {
                  'halite':
                      [0.0, 0.0, 500, 500, 500, 500, 0.0, 0.0,
                       386.014, 0.0, 500, 0.0, 0.0, 500, 0.0, 386.014,
                       0.0, 0.0, 13.146, 500, 500, 13.146, 0.0, 0.0,
                       77.234, 0.0, 1.195, 431.17, 230.141, 1.149, 0.0, 106.365,
                       57.389, 0, 1.195, 439.793, 317.635, 1.195, 0.0, 106.365,
                       0.0, 0.0, 12.888, 500, 500, 13.146, 0.0, 0.0,
                       386.014, 0.0, 500,
                       0.0, 0.0, 500, 0.0, 386.014, 0.0, 0.0, 500, 500, 500, 500, 0.0, 0.0],
                  'players': [
                      [3042, {'4-1': 33}, {'6-1': [24, 25], '9-1': [33, 0]}],
                      [5000, {}, {'0-2': [27, 267]}]
                  ],
                  'player': 0,
                  'step': 9}, 'status': 'DONE'}


class TestShipStateWrapper(TestCase):

    def setUp(self):
        obs = Observation(test_state['observation'])
        self.obs = obs
        self.ship_state_wrapper = ShipStateWrapper(radius=2, max_frames=2, map_size=8)

    def test_get_frame_ship_state_basic(self):
        ship_state = self.ship_state_wrapper.get_frame_ship_state_basic(obs=self.obs, current_player=self.obs.player)

        self.assertEqual(ship_state.shape, (8, 8))
        self.assertEqual(ship_state[3][3], -267.0)
        self.assertEqual(ship_state[3, 0], 25.0)

    def test_get_frame_shipyard_state_basic(self):
        ship_state = self.ship_state_wrapper.get_frame_shipyard_state_basic(obs=self.obs,
                                                                            current_player=self.obs.player)
        self.assertEqual(ship_state.shape, (8, 8))
        self.assertEqual(ship_state[4][1], 1)

    def test_get_basic_single_frame_complete_observation(self):
        complete_state = self.ship_state_wrapper.get_basic_single_frame_complete_observation(
            obs=self.obs,
            player=self.obs.player,
            spos=24
        )
        # 24 // 8 + 8 = 11
        # 24 % 8 + 8 = 8
        halite_state = complete_state[:, :, 0]
        ship_state = complete_state[:, :, 1]

        print('-----')
        print(complete_state.shape)
        print(complete_state.flatten().shape)

        self.assertEqual(complete_state.shape, (5, 5, 3))

        # asserting it is centered correctly
        self.assertEqual(halite_state[2, 2], 77.234)
        self.assertEqual(ship_state[2, 2], 25)

        # assert the edges
        self.assertEqual(halite_state[4, 4], 12.888)
        # assert the wraparound side
        self.assertEqual(halite_state[2, 1], 106.365)
