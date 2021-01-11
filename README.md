## Halite Simulation Game

Halite is a game developed by TwoSigma where players must mine and collect "halite" and deposit their halite at shipyards.

This project began when I stumbled upon this Kaggle competition.
https://www.kaggle.com/c/halite-iv-playground-edition

Earlier in the year another Halite competition was put forth. 
https://www.kaggle.com/c/halite

There are various notebooks and write-ups on winning strategies but I wanted to attempt using a reinforcement learning strategy.

## Reinforcement Learning

My approach was to use the Deep Q Learning algorithm.

For those new to Deep-Q learning, this graphic encapsulates the process.

![alt text](https://miro.medium.com/max/1120/1*zKvQWW05zfSaCfSEHviayA.png)

Source: [https://medium.com/@karan_jakhar/100-days-of-code-day-4-6fbc672171e4]


My next attempts will likely involve Actor-Critic or other gradient-based policies.

## V1: Kaggle Notebooks and Local Training

V1 was my initial attempt at running the training and analysis. It mainly involved uploading notebooks and code to the Kaggle compute environment.


## Halite Reinforcement Learning on GCP

This portion of the repo is dedicated to training the Halite reinforcement learning agent on the Google Cloud Platform.

This repo is based off of a combination of several GCP repositories/resources:

- https://cloud.google.com/blog/products/ai-machine-learning/deep-reinforcement-learning-on-gcp-using-hyperparameters-and-cloud-ml-engine-to-best-openai-gym-games
- https://cloud.google.com/ai-platform/docs/getting-started-keras#train_your_model_locally
- https://cloud.google.com/ai-platform/docs/getting-started-tensorflow-estimator

### Approach 1

1. Run Training
2. Pull trained weights into notebook environment
3. Load weights into new DQN
4. Perform analysis and watch what agent does.

### Approach 2

1. Run hyperparameter jobs
2. Select 2-3 best models
3. Perform steps 2-4 from Approach 1

