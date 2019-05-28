import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size.
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 3e-3         # learning rate of the actor 
LR_CRITIC = 3e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.batch_size = BATCH_SIZE

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
    
    def random_experience(self, state, action, reward, next_state, done):
        # Save randomly generated experience in PrioritizedReplayBuffer memory
        self.memory.random_experience(state, action, reward, next_state, done)
        
    def reset_experience_pointer(self):
        self.memory.reset_experience_pointer()
    
    def get_experience(self):
        
        if (self.memory.experience_size() > BATCH_SIZE):
            experiences = self.memory.sample()
            return experiences
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
        
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
    
    def critic_learn(self, states, actions, rewards, next_states, dones, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Apply Gradient Clipping when training the critic network
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
    
    def actor_learn(self, states): 
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def update_target_networks(self):
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05): 
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

    
class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-
    tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    experience_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity 
        
        # Generate a sumtree with all nodes values = 0 
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        # Size of tree = (2 * capacity - 1). Binary node has max 2 children, hence 2x size of leaf (capacity) - 1 (root node)
        
        # Number of leaf nodes (final nodes) that contains priority score for experiences
        self.tree = np.zeros(2 * capacity - 1)  #e.g. (2x4 - 1) = 7 nodes in the tree
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] 
        """
        
        # Contains the experiences ["state", "action", "reward", "next_state", "done"] (so the size of experience is capacity)
        self.experiences = np.zeros(capacity, dtype=object)
        
        self.lowest_level_right_tree_index = 2 * capacity - 2
        self.lowest_level_left_tree_index = self.lowest_level_right_tree_index - (capacity - 1) 
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data (fill the leaves from left to right)
    """
    def add(self, priority, experience):
        # For example, the first experience goes to (0 + 4 - 1 = 3) node
        tree_index = self.experience_pointer + self.capacity - 1
        
        """ tree:
             0
            / \
           0   0
          / \ / \
tree_index  0 0  0
        """
        
        # Update experience data frame
        self.experiences[self.experience_pointer] = experience
        
        # Update the leaf and propagate the change in priority throughout the tree
        self.update(tree_index, priority)
        
        # Add 1 to experience_pointer
        self.experience_pointer += 1
        
        if self.experience_pointer >= self.capacity:  # If exceed capacity, the first leaf node on the left will be overwritten
            self.experience_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        
        # Replace the former priority score in tree with new priority score
        self.tree[tree_index] = priority
        
        # Propagate the computed change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            At tree_index 6, we updated the priority score
            After which, tree_index 2 has to be updated
            Next tree_index to be updated = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (// i.e. integer division)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v): # v is the priority value
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
           /   \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        
        Create array of size (=capacity) to store experiences. 
        [0,1,2,3]
        
        Only the lowest level tree_index[-capacity:] will be populated with experience
        """
        parent_index = 0
        experience_index = 0
        leaf_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        experience_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.experiences[experience_index]
    
    @property 
    def total_priority(self):
        return self.tree[0] # Returns the root node

    
class PrioritizedReplayBuffer:
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01 # Hyperparameter to avoid some experiences with 0 probability of being taken
    PER_a = 0.6  # Hyperparameter to make a tradeoff between taking only experience with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.   # clipped abs error

    tree = SumTree(BUFFER_SIZE) # make tree a class variable to be shared among agents
    
    def __init__(self, buffer_size, batch_size, seed):
        # Initialising the tree 
        """
        The tree is composed of a sum tree that contains the priority scores at his leaf as well as a data array for experience
        Instead of deque, a simple array is used and overwritten when the memory is full.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    """
    Store a new experience in the sumtree
    Each new experience have a score of max_priority (it will be then improved when we use this exp to train our DDQN)
    """
    def add(self, state, action, reward, next_state, done):
        # Create experience object 
        e = self.experience(state, action, reward, next_state, done)
        
        # Find the max priority. "[-self.tree.capacity:]" return the 1st to last element in the data array
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0, the experience will never have a chance to be selected. Hence, set a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        # Set the max p for the new experience and store it to sumtree
        self.tree.add(max_priority, e)   

    
    def random_experience(self, state, action, reward, next_state, done):
        # Create experience object 
        e = self.experience(state, action, reward, next_state, done)
        
        self.tree.add(np.random.uniform(self.PER_b, 0.6), e)
        
    """
    - To sample a minibatch of k size, the range [0, priority_total] is divided into k ranges.
    - Next, a value is uniformly sampled from each range
    - Then, the experiences where the priority score corresponded to the sample values are retrieved from the sumtree.
    - Finally, the IS weights for each minibatch element is computed
    """
    def sample(self):
        # Create a sample array that will contains the minibatch
        exp_buffer = []
        
        b_idx, b_ISWeights = np.empty((self.batch_size,), dtype=np.int32), np.empty((self.batch_size, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / self.batch_size       # priority segment
        
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # increasing from 0.4 to max = 1
        
        # Calculating the max_weight
        if (np.min(self.tree.tree[-self.tree.capacity:])==0):
            p_min = self.absolute_error_upper / self.tree.total_priority
        else:
            p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
            
        max_weight = (p_min * self.batch_size) ** (-self.PER_b)
        
        
        for i in range(self.batch_size):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each priority value is retrieved
            """
            index, priority, experience = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(self.batch_size * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            exp_buffer.append(experience)
                
        states = torch.from_numpy(np.vstack([e.state for e in exp_buffer if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exp_buffer if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exp_buffer if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exp_buffer if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exp_buffer if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones, b_idx, b_ISWeights) 
    
    
    """
    Update the priorities on the sumtree after training
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

        
    def experience_size(self):
        """Return the current size of internal experience array."""
        return self.tree.experience_pointer
    
    def reset_experience_pointer(self):
        self.tree.experience_pointer = 0