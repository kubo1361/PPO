from agents import AgentPPO
from networks import network
from workers import Worker
from gym_wrapper import make_env


def train():
    # train
    actions = 5
    workers_len = 25
    stack = 4
    steps = 5
    epochs = 5
    observations_per_epoch = 5
    name = "test_pacman"
    id = 0

    goal_episodes = 20000
    start_episode=0
    start_steps=0
    start_score=0

    agent = AgentPPO(name=name, model=network(actions), id=id)

    workers = []
    for id_w in range(workers_len):
        env = make_env('MsPacman-v0', stack)
        env.seed(id_w)
        w = Worker(id_w, env, agent, print_score=False)
        workers.append(w)

    path = 'models/{name}/{name}_{id}_{suffix}.pt'.format(name = name, id=id, suffix='best')
    agent.load_model(path)
    agent.train(workers=workers, episodes=goal_episodes, steps=steps, 
            epochs=epochs, observations_per_epoch=observations_per_epoch, 
            start_episode=start_episode, start_score=start_score, 
            start_steps=start_steps)

    

if __name__ == '__main__':
    train()
    