from agents import AgentPPO
from networks import network
from workers import Worker
from gym_wrapper import make_env


def train():
    actions = 5
    workers_len = 20
    iterations = 10000
    stack = 4
    steps = 10

    agent = AgentPPO(name="test_pacman", model=network(actions),
                    lr=0.0001, id=0)

    workers = []
    for id in range(workers_len):
        env = make_env('MsPacman-v0', stack)
        env.seed(id)
        w = Worker(id, env, agent, print_score=True)
        workers.append(w)


    # path = 'models/final/final_3_10000_a2c.pt'
    # agent.load_model(path)
    agent.train(workers=workers, iterations=iterations, steps=steps)

if __name__ == '__main__':
    train()
