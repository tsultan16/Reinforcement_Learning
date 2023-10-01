from abc import ABC, abstractmethod
import numpy as np
import random


# first, define an interface for an MDP
class MDP:
    @abstractmethod
    def get_states(self):
        pass   

    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_terminal_states(self):
        pass    

    @abstractmethod
    def get_actions(self, state):
        pass

    @abstractmethod
    def get_transitions(self, state, action):
        pass

    @abstractmethod
    def get_rewards(self, state, action, next_state):
        pass

    @abstractmethod
    def is_terminal(self, state):
        pass

    @abstractmethod
    def get_discount_factor(self):
        pass
    
    @abstractmethod
    def execute(self):
        pass



# now, implement the gridworld problem MDP
class GridWorld(MDP):

    def __init__(self, discount_factor=0.9, noise=0.1) :

        # initialise the set of all possible states, in this case tuples (x,y) of all grid cells, excluding walls
        # will use the example from lectures and tute week 7
        self.width = 4
        self.height = 3
 
        # specify walls/blocked states
        self.walls = [(1,1)]
 
        # specify terminal states
        self.terminal_states=[(3,1),(3,2)] 

        # create a dummy terminal state which is the successor to all terminal states
        self.exit = (-1,-1)

        # specify initial state
        self.initial_state = (0,0)

        # list of all allowed states
        self.states = [self.exit]
        for x in range(self.width):
            for y in range(self.height):
                if (x,y) not in self.walls:
                   self.states.append((x,y))

        # specify probability of splipping
        self.noise = noise

        # specify/enumerate the actions
        self.terminate = 0
        self.up = 1
        self.down = 2
        self.left = 3
        self.right = 4
        self.action_names = {self.up : 'up', self.down : 'down', self.left: 'left', self.right : 'right', self.terminate: 'end'}
        
        # set the discount factor
        self.gamma = discount_factor

        # specify rewards
        self.rewards =  {(3,1) : -1, (3,2) : 1}

        # specify action cost 
        self.action_cost = 0.0

        # initialize Q-function and value function
        self.Q = {}
        self.V = {}
        self.Vtemp = {}
        for state in (self.states):
            self.V[state] = 0.0
            self.Vtemp[state] = 0.0
            actions = self.get_actions(state)
            for action in actions:
                self.Q[(state, action)] = 0.0     


    def get_states(self):
        return self.states   


    def get_initial_state(self):
        return self.initial_state


    def get_terminal_states(self):
        return self.terminal_states


    def get_actions(self, state):

        actions = [self.terminate, self.up, self.down, self.left, self.right]
        if state is None:
            return actions
        
        valid_actions = []
        for action in actions:
            for (next_state, probability) in self.get_transitions(state, action):
                if probability > 0:
                    valid_actions.append(action)
                    break
                
        return valid_actions        

    
    # for given state-action pair, returns possible successor states along with their corresponding transition probabilities
    def get_transitions(self, state, action):
        
        # probability of not slipping
        straight = 1 - 2*self.noise

        transitions = []
        (x,y) = state


        # if in terminal state or exit state, then only the 'terminate' action is available        
        if state in self.terminal_states or state == self.exit:
            if action == self.terminate:
                return [(self.exit, 1.0)] 
            else:
                return []

        # for non-terminal states only up,down,left,right actions avaialable
        elif action == self.up:
            transitions.append(self.valid_add(state, (x,y+1), straight))
            transitions.append(self.valid_add(state, (x-1,y), self.noise))
            transitions.append(self.valid_add(state, (x+1,y), self.noise))

        elif action == self.down:
            transitions.append(self.valid_add(state, (x,y-1), straight))
            transitions.append(self.valid_add(state, (x-1,y), self.noise))
            transitions.append(self.valid_add(state, (x+1,y), self.noise))

        elif action == self.left:
            transitions.append(self.valid_add(state, (x-1,y), straight))
            transitions.append(self.valid_add(state, (x,y-1), self.noise))
            transitions.append(self.valid_add(state, (x,y+1), self.noise))

        elif action == self.right:
            transitions.append(self.valid_add(state, (x+1,y), straight))
            transitions.append(self.valid_add(state, (x,y-1), self.noise))
            transitions.append(self.valid_add(state, (x,y+1), self.noise))

        return transitions
    

    def get_rewards(self, state, action, next_state):
        # transitioning to the exit state will earn the non-zero rewards
        if (state in self.terminal_states) and (next_state == self.exit):
            reward = self.rewards[state]
        else:
            reward = self.action_cost

        return reward    


    def is_exit(self, state):
        return (state == self.exit)    


    def get_discount_factor(self):
        return self.gamma  

    
    def execute(self, state, action):
        # get all transitions
        transitions = self.get_transitions(state, action)
        states = [tr[0] for tr in transitions]
        probs = [tr[1] for tr in transitions]
        # sample from the transitions to get the next state state
        next_state = random.choices(states, weights=probs, k=1)[0]
        reward = self.get_rewards(state, action, next_state)
        return (next_state, reward)
    

    def valid_add(self, state, next_state, probability):

        # check if next state is a wall
        if next_state in self.walls:
            return (state, probability)

        # check if next state is off grid
        (x,y) = next_state
        if (x>=0 and x<=(self.width-1) and y>=0 and y<=(self.height-1)):
            return (next_state, probability)
        else:
            return (state, probability)


    def update_Q(self, state, action, Qnew):
        self.Q[(state, action)] = Qnew
    
    
    def update_V(self):
        # copy from Vtemp
        for state in self.states:
            self.V[state] = self.Vtemp[state]


    def update_V_from_Q(self):
        for state in self.states:
            actions = self.get_actions(state)
            self.V[state] =  max([self.Q[(state, action)] for action in actions])


    def extract_policy(self):
        # extract a policy using the Q function
        policy = {}
        for state in self.states:
            actions = self.get_actions(state)
            Qs = {}
            for action in actions:
                Qs[action] = self.Q[(state, action)]    

            policy[state] = self.action_names[max(Qs, key=lambda k:Qs[k])]
            
        return policy


class CliffWorld(MDP):

    def __init__(self, discount_factor=0.9, noise=0.01) :

        # initialise the set of all possible states, in this case tuples (x,y) of all grid cells, excluding walls
        # will use the example from lectures and tute week 7
        self.width = 6
        self.height = 4
 
        # specify walls/blocked states
        self.walls = []
 
        # specify terminal states
        self.terminal_states = [(1,0),(2,0),(3,0),(4,0),(5,0)] 

        # create a dummy terminal state which is the successor to all terminal states
        self.exit = (-1,-1)

        # specify initial state
        self.initial_state = (0,0)

        # list of all allowed states
        self.states = [self.exit]
        for x in range(self.width):
            for y in range(self.height):
                if (x,y) not in self.walls:
                   self.states.append((x,y))

        # specify probability of splipping
        self.noise = noise

        # specify/enumerate the actions
        self.terminate = 0
        self.up = 1
        self.down = 2
        self.left = 3
        self.right = 4
        self.action_names = {self.up : 'up', self.down : 'down', self.left: 'left', self.right : 'right', self.terminate: 'end'}
        
        # set the discount factor
        self.gamma = discount_factor

        # specify rewards for transitioning into terminal states
        self.rewards =  {(1,0) : -5.0, (2,0) : -5.0, (3,0) : -5.0, (4,0) : -5.0, (5,0) : 0.0}

        # specify action cost 
        self.action_cost = -0.05

        # empty list for storing the discounted reward at each step opf the episode
        self.episode_discounted_rewards = []

        # initialize Q-function and value function
        self.Q = {}
        self.V = {}
        self.Vtemp = {}
        for state in (self.states):
            self.V[state] = 0.0
            self.Vtemp[state] = 0.0
            actions = self.get_actions(state)
            for action in actions:
                self.Q[(state, action)] = 0.0     


    def get_states(self):
        return self.states   


    def get_initial_state(self):
        return self.initial_state


    def get_terminal_states(self):
        return self.terminal_states


    def get_actions(self, state):

        actions = [self.terminate, self.up, self.down, self.left, self.right]
        if state is None:
            return actions
        
        valid_actions = []
        for action in actions:
            for (next_state, probability) in self.get_transitions(state, action):
                if probability > 0:
                    valid_actions.append(action)
                    break
                
        return valid_actions        

    
    # transition function
    def get_transitions(self, state, action):
        
        # probability of not slipping
        straight = 1 - 2*self.noise

        transitions = []
        (x,y) = state

        # if in terminal state or exit state, then only the 'terminate' action is available        
        if state in self.terminal_states or state == self.exit:
            if action == self.terminate:
                return [(self.exit, 1.0)] 
            else:
                return []
            
        # for non-terminal states, only up,down,left,right actions avaialable
        elif action == self.up:
            transitions.append(self.valid_add(state, (x,y+1), straight))
            transitions.append(self.valid_add(state, (x-1,y), self.noise))
            transitions.append(self.valid_add(state, (x+1,y), self.noise))

        elif action == self.down:
            transitions.append(self.valid_add(state, (x,y-1), straight))
            transitions.append(self.valid_add(state, (x-1,y), self.noise))
            transitions.append(self.valid_add(state, (x+1,y), self.noise))

        elif action == self.left:
            transitions.append(self.valid_add(state, (x-1,y), straight))
            transitions.append(self.valid_add(state, (x,y-1), self.noise))
            transitions.append(self.valid_add(state, (x,y+1), self.noise))

        elif action == self.right:
            transitions.append(self.valid_add(state, (x+1,y), straight))
            transitions.append(self.valid_add(state, (x,y-1), self.noise))
            transitions.append(self.valid_add(state, (x,y+1), self.noise))

        return transitions
    

    # reward function
    def get_rewards(self, state, action, next_state):
        if next_state in self.terminal_states:
            reward = self.rewards[next_state]
        elif next_state == self.exit:
            reward = 0.0                
        else:
            reward = self.action_cost

        # store discounted reward for the step
        step = len(self.episode_discounted_rewards)
        self.episode_discounted_rewards.append(reward * (self.gamma**step))

        return reward    


    def is_exit(self, state):
        return (state == self.exit)    


    def get_discount_factor(self):
        return self.gamma  

    
    def execute(self, state, action):
        # get all transitions
        transitions = self.get_transitions(state, action)
        states = [tr[0] for tr in transitions]
        probs = [tr[1] for tr in transitions]
        # sample from the transitions to get the next state state
        next_state = random.choices(states, weights=probs, k=1)[0]
        reward = self.get_rewards(state, action, next_state)
        return (next_state, reward)
    

    def valid_add(self, state, next_state, probability):

        # check if next state is a wall
        if next_state in self.walls:
            return (state, probability)

        # check if next state is off grid
        (x,y) = next_state
        if (x>=0 and x<=(self.width-1) and y>=0 and y<=(self.height-1)):
            return (next_state, probability)
        else:
            return (state, probability)


    def update_Q(self, state, action, Qnew):
        self.Q[(state, action)] = Qnew
    
    
    def update_V(self):
        # copy from Vtemp
        for state in self.states:
            self.V[state] = self.Vtemp[state]


    def update_V_from_Q(self):
        for state in self.states:
            actions = self.get_actions(state)
            self.V[state] =  max([self.Q[(state, action)] for action in actions])


    def extract_policy(self):
        # extract a policy using the Q function
        policy = {}
        for state in self.states:
            actions = self.get_actions(state)
            Qs = {}
            for action in actions:
                Qs[action] = self.Q[(state, action)]    

            policy[state] = self.action_names[max(Qs, key=lambda k:Qs[k])]
            
        return policy



# value iteration loop
def value_iteration(grid_world_mdp, num_iters, theta=0.001):
    states = grid_world_mdp.get_states()    
    for i in range(num_iters):
        print(f"Iteration# {i}")
        d = 0.0 
        for state in states:
            #print(f"state: {state}")
            actions = grid_world_mdp.get_actions(state)
            Qsa = [] 
            for action in actions:
                # compute Q value
                Qnew = 0.0
                for (next_state, p) in grid_world_mdp.get_transitions(state, action):
                    Qnew += p * (grid_world_mdp.get_rewards(state, action, next_state) + grid_world_mdp.gamma * grid_world_mdp.V[next_state])
                grid_world_mdp.update_Q(state, action, Qnew)
                Qsa.append(Qnew)
            Vnew = max(Qsa)    
            grid_world_mdp.Vtemp[state] = Vnew  
            # update delta        
            d = max(d, abs(Vnew - grid_world_mdp.V[state]))
        
        # update the value function
        grid_world_mdp.update_V()

        print("-----------------------")
        for y in range(grid_world_mdp.height-1, -1, -1):
            for x in range(grid_world_mdp.width):
                if (x,y) in grid_world_mdp.V:
                    print(f"{grid_world_mdp.V[(x,y)]: 0.2f}", end=' ')
                else:
                    print(f"{0.0: 0.2f}", end=' ')
            print("")       
        print("-----------------------")


        # stop value iteration if delta falls below threshold    
        if d < theta:
            break    

# Q-learner class
class QLearner:
    def __init__(self, mdp, alpha=0.1, epsilon=0.1) :
        self.mdp = mdp
        self.alpha = alpha
        self.epsilon = epsilon
        np.random.seed(2)


    def set_alpha(self, alpha):
        self.alpha = alpha


    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


    # epsilon-greedy multi-arm bandit
    def bandit(self, state):    
        # get all available actions
        actions = self.mdp.get_actions(state)
        randnum = np.random.random()
        
        # exploration
        if randnum < self.epsilon:
            action = random.choices(actions, k=1)[0]
        
        # exploitation
        else:
            # get Q values
            Qsa = {action:self.mdp.Q[(state, action)] for action in actions}
            # argmax to find best action
            action = max(Qsa, key=Qsa.get)        
        return action


    # Q-learning update
    def get_delta_Q(self, reward, Qold, state, next_state, next_action):
        # get estimated value for next state
        aprime = self.mdp.get_actions(next_state)
        Vsprime = max([self.mdp.Q[(next_state, action)] for action in aprime])
        delta = reward + self.mdp.gamma * Vsprime - Qold
        return self.alpha * delta  

    # SARSA update
    def get_delta_SARSA(self, reward, Qold, state, next_state, next_action):
        # get estimated value for next state
        Vsprime = self.mdp.Q[(next_state, next_action)] 
        delta = reward + self.mdp.gamma * Vsprime - Qold
        return self.alpha * delta  


    # Q-learner training loop
    def train(self, episodes=10, SARSA=False):
        episode_rewards = []
        for i in range(episodes):
            episode_len = 0
            accumulated_discounted_reward = 0.0
            # get initial state for the episode
            state = self.mdp.get_initial_state()
            # select an action using bandit
            action = self.bandit(state)
            # repeat until terminal state is reached
            while not self.mdp.is_exit(state):
                (next_state, reward) = self.mdp.execute(state, action)
                next_action = self.bandit(next_state)
                # update q value
                Qold = self.mdp.Q[(state, action)]
                if not SARSA:
                    Qnew = Qold + self.get_delta_Q(reward, Qold, state, next_state, next_action)
                else:
                    Qnew = Qold + self.get_delta_SARSA(reward, Qold, state, next_state, next_action)
                self.mdp.Q[(state,action)] = Qnew
                state = next_state
                action = next_action
                
                # accumulate discounted reward for the step
                accumulated_discounted_reward += reward * (self.mdp.gamma**episode_len)
                episode_len += 1

            # save total accumulated reward after episode is over 
            episode_rewards.append(accumulated_discounted_reward)
            print(f"Episode# {i}, length: {episode_len}, accumulated reward: {accumulated_discounted_reward}")
   
            # update the value function
            self.mdp.update_V_from_Q()

            print("-----------------------")
            for y in range(self.mdp.height-1, -1, -1):
                for x in range(self.mdp.width):
                    if (x,y) in self.mdp.V:
                        print(f"{self.mdp.V[(x,y)]: 0.2f}", end=' ')
                    else:
                        print(f"{0.0: 0.2f}", end=' ')
                print("")       
            print("-----------------------")

        return episode_rewards
