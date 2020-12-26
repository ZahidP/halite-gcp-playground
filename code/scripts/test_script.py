from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board, ShipAction, ShipyardAction, Observation


env = make("halite", configuration={"episodeSteps": 10, "size": 8})


env.run(["random", "random"])


observation = Observation(env.state[0]['observation'])


print(env.configuration)

print('HALITE')
print(observation.halite)

print('PLAYERS')
print(observation.players)