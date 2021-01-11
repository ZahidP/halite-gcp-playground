
## Halite Reinforcement Learning on GCP

This portion of the repo is dedicated to training the Halite reinforcement learning agent on the Google Cloud Platform.

This repo is based off of a combination of several GCP repositories/resources:

- https://cloud.google.com/blog/products/ai-machine-learning/deep-reinforcement-learning-on-gcp-using-hyperparameters-and-cloud-ml-engine-to-best-openai-gym-games
- https://cloud.google.com/ai-platform/docs/getting-started-keras#train_your_model_locally
- https://cloud.google.com/ai-platform/docs/getting-started-tensorflow-estimator

### Code Structure

A bulk of the code lives in the rl_on_gcp directory.

- At the top level we have the `setup.py` file, the `requirements.txt`, and shell scripts.
- The trainer/ directory contains everything needed to perform model training.
- The main task is under trainer/task.py under `_run`

This setup is in accordance with the best practices for packaging a GCP training application.

