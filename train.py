from agents import AgentPPO
from networks import network
from workers import Worker
from gym_wrapper import make_env


# Main training method
def train():
    # Initial parameters
    actions = 5
    workers_len = 25
    stack = 4
    steps = 5
    epochs = 5
    observations_per_epoch = 5
    name = "test_pacman"
    id = 0

    goal_episodes = 300000
    
    # Create agent
    agent = AgentPPO(name=name, model=network(actions), id=id)

    # Create workers and environments
    workers = []
    for id_w in range(workers_len):
        env = make_env('MsPacman-v0', stack)
        env.seed(id_w)
        w = Worker(id_w, env, agent, print_score=False)
        workers.append(w)
    
    # Load model (we trained with interruptions)
    model_path = 'models/{name}/{name}_{id}_{suffix}.pt'.format(name = name, id=id, suffix='218333')
    agent.load_model(model_path)
    
    progress_path = 'models/{name}/progress.json'.format(name = name)
    progress = agent.load_progress(progress_path)
    start_episode=progress['episode']
    start_steps=progress['average_steps']
    start_score=progress['average_score']
    learning_rate = progress['learning_rate']

    # Train
    agent.train(workers=workers, episodes=goal_episodes, steps=steps, 
            epochs=epochs, observations_per_epoch=observations_per_epoch, 
            start_episode=start_episode, start_score=start_score, 
            start_steps=start_steps, lr=learning_rate)    

if __name__ == '__main__':
    train()
    