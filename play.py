from gym_wrapper import transform_observation, make_env
import torch
import gym
import numpy as np
from networks import network
import torch.nn.functional as F
import time

import matplotlib.pyplot as plt


class Agent:
    def __init__(self, model):
        # init vars
        self.model = model
        self.actions_count = model.actions_count

        # device - define and cast
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def choose_action(self, observation):
        # add dimension (batch) to match (batch, layer, height, width), transfer to GPU
        observation = observation.unsqueeze(0).to(self.device).float()

        # we do not compute gradients when choosing actions, hence no_grad
        with torch.no_grad():
            _, _, actions = self.model(observation)

        # transfer to CPU after calculation
        actions = actions.cpu()

        # sort probabilities

        # return highest probability
        return actions[0].item()


def play():
    path = 'models/{name}/{name}_{id}_{suffix}.pt'.format(name = 'test_pacman', id=0, suffix='end')
    actions = 5
    agent = Agent(network(actions))
    agent.load_model(path)

    env = gym.make('MsPacman-v0')
    init = env.reset()
    init = transform_observation(init)
    done = False
    action_ai = 0
    observations = np.zeros((4, 80, 80), dtype=np.float32)
    switch = True
    while True:
        while not done:
            for _ in range(4):
                env.render()
                obs, _, done, _ = env.step(action_ai)
                switch = not switch
            observations[:-1] = observations[1:]
            observations[-1] = transform_observation(obs)
            action_ai = agent.choose_action(torch.from_numpy(observations))
            time.sleep(1 / 60)  # FPS
        env.reset()
        done = False


def play2():
    path = 'models/{name}/{name}_{id}_{suffix}.pt'.format(name = 'test_pacman', id=0, suffix='end')
    actions = 5
    agent = Agent(network(actions))
    agent.load_model(path)

    env = make_env('MsPacman-v0', 4)
    env.reset()
    done = False
    action_ai = 0
    score = 0
    while not done:
        obs, reward, done, _ = env.step(action_ai)

        action_ai = agent.choose_action(torch.from_numpy(obs))
        score += reward
        print(score)


def trans_obs():
    env = gym.make('MsPacman-v0')
    obs = env.reset()
    transformed = transform_observation(obs)
    transformed = transformed.reshape(1, 80, 80)

    plt.imshow(transformed[0], interpolation='nearest')
    #plt.imshow(obs, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    play()
    
    # trans_obs()