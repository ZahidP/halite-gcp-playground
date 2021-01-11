"""TODO(praneetdutta): DO NOT SUBMIT without one-line documentation for train
# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicableQQlaw or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

import pandas as pd
import tensorflow as tf
from google.cloud import storage
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Observation

from trainer.agent import Agent
from trainer.halite_agent import HaliteAgent
from trainer.halite_env import HaliteEnv
from trainer.opponent_agents.basic_agents import do_nothing_agent
from trainer.ship_state_wrapper import ShipStateWrapper
from trainer.shipyard_state_wrapper import ShipYardStateWrapper

SAVE_STEPS_FACTOR = 50
DISPLAY_RESULTS = 25


def hp_directory(model_dir):
    """If running a hyperparam job, create subfolder name with trial ID.
    If not running a hyperparam job, just keep original model_dir."""
    trial_id = json.loads(
        os.environ.get('TF_CONFIG', '{}')
    ).get('task', {}).get('trial', '')
    return os.path.join(model_dir, trial_id)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


# @tf.function
def _run(args):
    print('Running')

    env = make("halite", {'episodeSteps': 200}, debug=True)

    """
    Setup all of the hyperparameters and other inputs
    """
    try:
        learn_rate = args[0].learning_rate
        discount_factor = args[0].discount_factor
        batch_size = args[0].batch_size
        layer_1_dims_ship = args[0].layer_1_dims_ship
        layer_1_dims_shipyard = args[0].layer_1_dims_shipyard
        epsilon = args[0].epsilon
        if args[0].mode == 'Train':
            # Add timestamp; important for HP tuning so models don't clobber each
            # other.
            model_dir = hp_directory(args[0].model_dir)
        else:
            model_dir = args[0].model_dir
        writer = tf.summary.create_file_writer(os.path.join(model_dir, 'metrics'))
    except Exception as e:
        print('Failure parsing args')
        print(e)
        raise e
    try:

        """
        Extra helpers needed for halite
        """

        ship_state_wrapper = ShipStateWrapper(
            radius=5,
            max_frames=2,
            map_size=env.configuration['size']
        )
        shipyard_state_wrapper = ShipYardStateWrapper(
            radius=5,
            max_frames=1,
            map_size=env.configuration['size']
        )

        ship_agent = Agent(
            alpha=learn_rate,
            gamma=discount_factor,
            n_actions=6,
            batch_size=batch_size,
            epsilon=epsilon,
            input_dims=ship_state_wrapper.state_size,
            fc1_dims=layer_1_dims_ship
        )

        shipyard_agent = Agent(
            alpha=learn_rate,
            gamma=discount_factor,
            n_actions=2,
            batch_size=batch_size,
            epsilon=epsilon,
            input_dims=shipyard_state_wrapper.state_size,
            fc1_dims=layer_1_dims_shipyard
        )

        halite_env = HaliteEnv(
            environment=env,
            opponents=[do_nothing_agent],
            ship_state_wrapper=ship_state_wrapper,
            shipyard_state_wrapper=shipyard_state_wrapper,
            ship_reward_type='total_halite'
        )

        halite_agent = HaliteAgent(
            ship_agent=ship_agent,
            shipyard_agent=shipyard_agent,
            configuration=env.configuration,
            halite_env=halite_env,
            ship_state_wrapper=ship_state_wrapper,
            shipyard_state_wrapper=shipyard_state_wrapper
        )
    except Exception as e:
        print('Failed setting up Halite Utilities')
        print(e)
        raise e

    print("STARTING...")

    """
    Loading previous model if possible
    """

    if not os.path.exists(model_dir) and args[0].save_model:
        os.makedirs(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'results')) and args[0].save_model:
        os.makedirs(os.path.join(model_dir, 'results'))
    print("MODEL WILL BE STORED AT: ", model_dir)

    if args[0].mode != 'Train':
        trained_model_path = args[0].load_model
        try:
            ship_agent.load_weights(trained_model_path)
        except:
            print('{} is not a valid .h5 model.'.format(trained_model_path))

    episode_reward, episode_number, done = 0, 0, False

    if done or args[0].mode != 'Train':
        state = env.reset()
    else:
        state, board = halite_env.reset()

    print('Beginning training loop')

    episode_results = []

    for curr_step in range(args[0].steps):

        raw_observation = state
        step_observation = Observation(raw_observation)

        """
        If we are not in training mode
        
            episode_reward = agent.play(env, model_dir, args[0].mode)
            print('CURRENT STEP: {}, EPISODE_NUMBER: {}, EPISODE REWARD: {},'
                  'EPSILON: {}'.format(curr_step, episode_number, episode_reward,
                                       eta))
            episode_run = False
        
        """

        """
        ==========
        TRAINING
        ==========
        """
        if args[0].mode == 'Train':
            # not sure how this step works
            # eta = anneal_exploration(eta, curr_step, args[0].steps / 10.0,
            #                          args[0].start_train, args[0].init_eta,
            #                          args[0].min_eta, 'linear')

            """
            Choose Actions.
            
            We will need an extra step here due to the simulation effect.
            
            if eta > np.random.rand() or curr_step < args[0].start_train:
                action = env.action_space.sample()
            else:
                action = agent.predict_action(state)
            """

            raw_observation, shipyard_simulated_step_memory = halite_agent.get_moves_for_all_shipyards(
                raw_observation=raw_observation,
                step_observation=step_observation,
                episode_number=episode_number,
                step_number=curr_step
            )

            raw_observation, ship_simulated_step_memory = halite_agent.get_moves_for_all_ships(
                raw_observation=raw_observation,
                step_observation=step_observation,
                episode_number=episode_number,
                step_number=curr_step
            )

            actions_for_step = {}

            for id_, action in halite_agent.actions_for_step.items():
                actions_for_step[id_] = action

            """
            Then we take the step.
            """

            next_state, reward, done = halite_env.step(actions=actions_for_step)

            """
            Add to the replay buffer.
            
             next_state, reward, done, info = env.step(action)
            Buffer.add_exp([state, next_state, reward, action, done])
            ready_to_update_model = curr_step > args[0].start_train and len(
                Buffer.buffer) > Buffer.min_size
            
            """

            halite_agent.learn(
                observation=next_state,
                shipyard_simulated_step_memory=shipyard_simulated_step_memory,
                ship_simulated_step_memory=ship_simulated_step_memory,
                episode_number=episode_number,
                step_number=curr_step,
                terminal=done
            )

            if (curr_step % SAVE_STEPS_FACTOR) == 0:
                ship_model_path_full = os.path.join(model_dir, 'ship_agent/')

                try:
                    print(f'Saving model at {ship_model_path_full}, Step: {curr_step} - Episode: {episode_number}')
                    ship_agent.save_weights(
                        model_path=ship_model_path_full
                    )
                    shipyard_agent.save_weights(
                        model_path=os.path.join(model_dir, 'shipyard_agent/')
                    )
                except Exception as e:
                    print('Failed to save weights')
                    print(e)
                    raise e

            scores = [vals[0] for vals in next_state.players]

            episode_results.append(scores)

            opponent = 0 if next_state.player else 1
            player_score = next_state.players[next_state.player][0]
            player_won = player_score > next_state.players[opponent][0]

            if (curr_step % DISPLAY_RESULTS) == 0:
                print(f'Scores {scores}')
                with writer.as_default():
                    tf.summary.scalar(name="SCORE", data=player_score, step=curr_step)
                    tf.summary.scalar(name="OPP_SCORE", data=next_state.players[opponent][0], step=curr_step)
                    writer.flush()

            """
            Update Model if conditions are met
            
            if ready_to_update_model:
                exp_state, exp_next_state, exp_reward, exp_action, exp_done = Buffer.sample_experiences(
                    args[0].batch_size)
                agent.batch_train(exp_state, exp_next_state, exp_reward, exp_action,
                                  exp_done, target_network, args[0].Q_learning)
                if curr_step % args[0].update_target == 0:
                    target_network.set_weights(agent.model.get_weights())
                if curr_step % (SAVE_STEPS_FACTOR *
                                args[0].update_target) == 0 and args[0].save_model:
            """

            """
            Save model if desired
                    models.save_model(
                        agent.model,
                        model_dir + 'model_' + str(episode_number) + '_.h5'
                    )
            """
            state = next_state

            # Resets state
            if done:
                print('Game done')
                with writer.as_default():
                    tf.summary.scalar(name="GAME_RESULT", data=player_won, step=curr_step)
                    writer.flush()
                print('Wrote summary stats')

                with open(os.path.join(model_dir, 'results', f"{episode_number}_results.csv"), 'w') as f:
                    pd.DataFrame(episode_results).to_csv(f)
                    f.close()

                episode_number += 1
                if args[0].mode != 'Train':
                    episode_run = True
                    state = env.reset()
                else:
                    state, board = halite_env.reset()
                done = False


def _parse_arguments(argv):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--frame_stack',
        help='Number of frames to stack ',
        type=float,
        default=4)
    parser.add_argument(
        '--steps',
        help='Number of steps for the agent to play the game',
        type=int,
        default=400)
    parser.add_argument(
        '--update_target',
        help='Number of steps after which to update the target network',
        type=int,
        default=1000)
    parser.add_argument(
        '--buffer_size',
        help='Size of the experience buffer',
        type=int,
        default=20000)
    parser.add_argument(
        '--mode',
        help='Whether we are training the agent or playing the game',
        type=str,
        default="Train")
    parser.add_argument(
        '--init_eta', help='Epsilon for taking actions', type=float, default=0.95)
    parser.add_argument(
        '--model_dir',
        help='Directory where to save the given model',
        type=str,
        default='models/')
    parser.add_argument(
        '--save_model',
        help='Whether to save the model',
        type=bool,
        default=False)
    parser.add_argument(
        '--batch_size',
        help='Batch size for sampling and training model',
        type=int,
        default=32)
    parser.add_argument(
        '--learning_rate',
        help='Learning rate for for agent and target network',
        type=float,
        default=0.00025)
    parser.add_argument(
        '--layer_1_dims_ship',
        help='Layer 1 dimesions for ship agent',
        type=float,
        default=64)
    parser.add_argument(
        '--layer_1_dims_shipyard',
        help='Layer 1 dimesions for shipyard agent',
        type=float,
        default=64)
    # layer_1_dims_ship
    parser.add_argument(
        '--Q_learning',
        help='Type of Q Learning to be implemented',
        type=str,
        default="Double")
    parser.add_argument(
        '--epsilon',
        help='Starting value of epsilon', type=float,
        default=0.99)
    parser.add_argument(
        '--discount_factor',
        help='Discount Factor for TD Learning',
        type=float,
        default=0.95)
    parser.add_argument(
        '--load_model', help='Loads the model', type=str, default=None)
    return parser.parse_known_args(argv)


def main():
    args = _parse_arguments(sys.argv[1:])
    _run(args)


if __name__ == '__main__':
    main()
