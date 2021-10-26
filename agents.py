import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from torch.utils.tensorboard import SummaryWriter


class AgentPPO:
    def __init__(self, name, model, gamma=0.99, lr=0.0001, beta_entropy=0.01, critic_loss_coef=0.5, grad_clip=0.1, epsilon=0.2, lr_decay=1e-7, id=0):
        # init vars
        self.model = model
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.critic_loss_coef = critic_loss_coef
        self.grad_clip = grad_clip
        self.epsilon = epsilon
        self.lr = lr
        self.lr_decay = lr_decay

        # device - define and cast
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', self.device)
        self.model.to(self.device)

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # create vars for tracking progress and identification
        # used by workers
        self.average_score = []
        self.average_steps = []
        self.episodes = 0

        # for convenience sake
        self.best_avg = 0

        # identification
        self.name = name
        self.id = id

        # create folders for models and logs
        self.writer = SummaryWriter('runs/' + self.name + '/' + str(self.id))
        self.model_path = 'models/' + self.name + '/'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)



    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def train(self, workers, iterations, steps, epochs=4, observations_per_epoch=4):
        self.model.train()
        
        # initial variables
        self.average_score = []
        self.average_steps = []
        new_observations = []
        self.best_avg = 0
        self.episodes = 0 
        len_workers = len(workers)

        all_epochs_per_iteration = steps * epochs
        index_range = torch.arange(all_epochs_per_iteration)

        # initial observations
        for worker in workers:
            new_observations.append(torch.from_numpy(worker.reset()).float())
        new_observations = torch.stack(new_observations).to(self.device)

        self.writer.add_graph(self.model, new_observations)

        for iteration in range(iterations):
            iter_critic_values = torch.zeros([all_epochs_per_iteration, len_workers, 1]).type(torch.cuda.FloatTensor)
            iter_actor_log_probs = torch.zeros([all_epochs_per_iteration, len_workers, 1]).type(torch.cuda.FloatTensor)
            iter_actions = torch.zeros([all_epochs_per_iteration, len_workers, 1]).type(torch.cuda.FloatTensor)
            iter_rewards = torch.zeros([all_epochs_per_iteration, len_workers, 1]).type(torch.cuda.FloatTensor)
            iter_not_terminated = torch.ones([all_epochs_per_iteration, len_workers, 1]).type(torch.cuda.FloatTensor)
            raw_advantages = torch.zeros([all_epochs_per_iteration, len_workers, 1]).type(torch.cuda.FloatTensor)
            diff_advantages = torch.zeros([all_epochs_per_iteration, len_workers, 1]).type(torch.cuda.FloatTensor)
            old_observations = []


            for epoch in range(all_epochs_per_iteration):
                # first forward pass with fresh observation
                with torch.no_grad():
                    epoch_actor, epoch_critic, epoch_actor_actions = self.model(new_observations)

                # after use it becomes old
                old_observations.append(new_observations)
                new_observations = []

                # epoch logarithmic probabilities
                epoch_log_probs = F.log_softmax(epoch_actor, dim=-1)
                epoch_log_policy = epoch_log_probs.gather(1, epoch_actor_actions)

                # reset epoch specific variables
                epoch_rewards = torch.zeros([len_workers, 1])
                epoch_not_terminated = torch.ones(
                    [len_workers, 1], dtype=torch.int8)

                # generate new observations for next pass and rewards for actual actions
                for worker in range(len_workers):
                    # Apply actions to workers enviroments
                    worker_observation, epoch_rewards[worker, 0], worker_terminated = workers[worker].step(
                        epoch_actor_actions[worker].item())

                    # reset terminated workers
                    if worker_terminated:
                        epoch_not_terminated[worker, 0] = 0
                        worker_observation = workers[worker].reset()

                    # append new observations
                    new_observations.append(torch.from_numpy(
                        worker_observation).float())

                # update iteration specific variables
                iter_critic_values[epoch] = epoch_critic
                iter_actor_log_probs[epoch] = epoch_log_policy
                iter_actions[epoch] = epoch_actor_actions
                iter_rewards[epoch] = epoch_rewards
                iter_not_terminated[epoch] = epoch_not_terminated
                new_observations = torch.stack(new_observations).to(self.device)


            # second forward pass for critic values on new observations
            with torch.no_grad():
                _, new_critic_values, _ = self.model(new_observations)

            # compute advantage - we compute advantage backwards through all epochs
            # with their respective critic values for each epoch
            for epoch in reversed(range(all_epochs_per_iteration)):
                new_critic_values = iter_rewards[epoch] + \
                    (self.gamma * new_critic_values * iter_not_terminated[epoch])

                diff_advantages[epoch] = new_critic_values - iter_critic_values[epoch]
                raw_advantages[epoch] = new_critic_values 

            # standard score normalization of advantage
            raw_advantages = (raw_advantages - torch.mean(raw_advantages)) / \
                (torch.std(raw_advantages) + 1e-5)
            
            diff_advantages = (diff_advantages - torch.mean(diff_advantages)) / \
                (torch.std(diff_advantages) + 1e-5)

            for epoch in range(epochs):
                index = index_range[torch.randperm(all_epochs_per_iteration)].flatten(start_dim=0)
                for batch in range(observations_per_epoch):
                    epoch_index = index[epoch]
                    epoch_observation = old_observations[epoch_index]

                    epoch_actor, epoch_critic, _ = self.model(epoch_observation)
                    
                    new_epoch_log_probs = F.log_softmax(epoch_actor, dim=-1)

                    new_epoch_log_policy = new_epoch_log_probs.gather(1, iter_actions[epoch_index].type(torch.int64))     
                    
                    epoch_probs = F.softmax(epoch_actor, dim=-1)
                    epoch_entropies = (
                        new_epoch_log_probs * epoch_probs).sum(1, keepdim=True)

                    new_advantage = raw_advantages[epoch_index] - epoch_critic

                    ratio = torch.exp(new_epoch_log_policy - iter_actor_log_probs[epoch_index])
                    epoch_advangate = diff_advantages[epoch_index]
                    clip = torch.clamp(ratio, min=1 - self.epsilon, max=1 + self.epsilon)

                    ratio_adv = ratio * epoch_advangate
                    clip_adv = clip * epoch_advangate

                    actor_loss = - torch.min(ratio_adv, clip_adv).mean()
                    # actor_loss = - (iter_actor_log_probs * advantages_detached).mean()
                    critic_loss = (new_advantage**2).mean() * self.critic_loss_coef
                    entropy_loss = (epoch_entropies.mean() * self.beta_entropy)

                    # print("actor_loss: ", actor_loss.item(), "critic_loss: ", critic_loss.item(), "entropy_loss: ", entropy_loss.item())
            
                    # clear gradients
                    self.optimizer.zero_grad()

                    # calculate final loss
                    loss = actor_loss + critic_loss + entropy_loss

                    # backward pass with our total loss https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
                    loss.backward()

                    # gradient clipping for exploding gradients https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    # optimizer step
                    self.optimizer.step()

            self._write_stats(iteration, iterations, actor_loss, critic_loss, entropy_loss, loss)            
        self.writer.close()


    def _write_stats(self, iteration, iterations, actor_loss, critic_loss, entropy_loss, loss):
         # stats
        if iteration % (iterations / 1000) == 0 and iteration > 0:

            # average for last 100 scores
            avg_score = np.average(self.average_score[-100:])
            avg_steps = np.average(self.average_steps[-100:])

            # lower learning rate
            self.lr = max(self.lr - self.lr_decay, 1e-7)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            # save model on new best average score
            if avg_score > self.best_avg:
                self.best_avg = avg_score
                print('Saving model, best score is: ', self.best_avg)
                model_filename = (
                    self.model_path + self.name + '_' + str(self.id) + '.pt')
                self.save_model(model_filename)

        if iteration % (iterations / 100) == 0 and iteration > 0:

            # display informations
            print('iteration: ', iteration, '\tepisodes: ',
                    self.episodes, '\taverage steps: ', avg_steps, '\taverage score: ', avg_score)

            # write to tensorboard
            self.writer.add_scalar('Episodes per iteration',
                    self.episodes, iteration)

            self.writer.add_scalar('Actor loss per iteration',
                    actor_loss.item(), iteration)

            self.writer.add_scalar('Critic loss per iteration',
                    critic_loss.item(), iteration)

            self.writer.add_scalar('Entropy loss per iteration',
                    entropy_loss.item(), iteration)

            self.writer.add_scalar('Final loss per iteration',
                    loss.item(), iteration)

            self.writer.add_scalar('Average steps per 100 episodes',
                    avg_steps, self.episodes)

            self.writer.add_scalar('Average score per 100 episodes',
                    avg_score, self.episodes)

        if iteration % (iterations / 10) == 0 and iteration > 0:
            continuous_save_model_filename = (
                self.model_path + self.name + '_' + str(self.id) + '_' + str(iteration) + '.pt')
            self.save_model(continuous_save_model_filename)








                