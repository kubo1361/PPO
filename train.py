from agents import AgentPPO
from networks import network
from workers import Worker
# from gym_wrapper import make_env
import gym

CONTINUE = False

def train():
    # train
    actions = 2
    workers_len = 25
    # stack = 4
    steps = 5
    epochs = 5
    observations_per_epoch = 5
    name = "cartpole"
    id = 0

    start_episode=0
    start_steps=0
    start_score=0
    learning_rate =0.0001

    goal_episodes = 15000
    
    agent = AgentPPO(name=name, model=network(actions), id=id)

    workers = []
    for id_w in range(workers_len):
        # env = make_env('CartPole-v1', stack)
        env = env = gym.make("CartPole-v1")
        env.seed(id_w)
        w = Worker(id_w, env, agent, print_score=False)
        workers.append(w)
    

    
    
    if CONTINUE:
        model_path = 'models/{name}/{name}_{id}_{suffix}.pt'.format(name = name, id=id, suffix='218333')
        agent.load_model(model_path)
        progress_path = 'models/{name}/progress.json'.format(name = name)
        progress = agent.load_progress(progress_path)
        start_episode=progress['episode']
        start_steps=progress['average_steps']
        start_score=progress['average_score']
        learning_rate = progress['learning_rate']

    agent.train(workers=workers, episodes=goal_episodes, steps=steps, 
            epochs=epochs, observations_per_epoch=observations_per_epoch, 
            start_episode=start_episode, start_score=start_score, 
            start_steps=start_steps, lr=learning_rate)    

if __name__ == '__main__':
    train()
    