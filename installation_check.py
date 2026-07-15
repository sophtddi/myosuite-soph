from myosuite.utils import gym
env = gym.make('myoChallengeOslRunFixed-v0')
env.reset()
print("OK, env chargé sans problème")