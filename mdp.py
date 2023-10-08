from abc import ABC, abstractmethod
import numpy as np
import random, math
import matplotlib.pyplot as plt


# first, define an interface for an MDP
class MDP(ABC):
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
    def is_exit(self, state):
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

        # specify goal
        self.goal = (3,2)

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


    def get_states(self):
        return self.states   


    def get_initial_state(self):
        return self.initial_state


    def get_terminal_states(self):
        return self.terminal_states


    def get_actions(self, state=None):

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

        '''
        if next_state in self.terminal_states:
            reward = self.rewards[next_state]
        elif next_state == self.exit:
            reward = 0.0                
        else:
            reward = self.action_cost
        '''

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


class CliffWorld(MDP):

    def __init__(self, discount_factor=0.9, noise=0.01, withQTable=True) :

        # initialise the set of all possible states, in this case tuples (x,y) of all grid cells, excluding walls
        # will use the example from lectures and tute week 7
        self.width = 6
        self.height = 4
 
        # specify walls/blocked states
        self.walls = []
 
        # specify terminal states
        self.terminal_states = [(1,0),(2,0),(3,0),(4,0),(5,0)] 

        # specify goal
        self.goal = (5,0)

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
        self.rewards =  {(1,0) : -50.0, (2,0) : -50.0, (3,0) : -50.0, (4,0) : -50.0, (5,0) : 5.0}

        # specify action cost 
        self.action_cost = -0.05

        # empty list for storing the discounted reward at each step opf the episode
        self.episode_discounted_rewards = []


    def get_states(self):
        return self.states   


    def get_initial_state(self):
        return self.initial_state


    def get_terminal_states(self):
        return self.terminal_states


    def get_actions(self, state=None):

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


# a Q-function interface
class QFunction(ABC):
    @abstractmethod
    def update(self, state, action, delta):
        pass
    
    @abstractmethod
    def evaluate(self, state, action):
        pass

    # returns argmax action and Q-value for given state
    def get_maxQ(self, state, actions):
        argmax_q = None
        max_q = float("-inf")
        for action in actions:
            value = self.evaluate(state, action)
            if max_q < value:
                max_q = value
                argmax_q = action

        return (argmax_q, max_q)


    # extract a policy using the Q function
    def extract_policy(self, mdp):
        policy = {}
        for state in mdp.states:
            actions = mdp.get_actions(state)
            Qs = {}
            for action in actions:
                Qs[action] = self.evaluate(state, action)    

            policy[state] = mdp.action_names[max(Qs, key=lambda x:Qs[x])]
            
        return policy


# a Q-table base class that derives from Q function interface
class QTable(QFunction):
    def __init__(self, mdp, default=0.0):
        self.mdp = mdp
        self.default = default
        # initialize Q-function and value function 
        self.Q = {}
        self.V = {}
        self.Vtemp = {}
        
        for state in (self.mdp.states):
            self.V[state] = default
            self.Vtemp[state] = default
            actions = self.mdp.get_actions()
            for action in actions:
                self.Q[(state, action)] = default

    def reset(self):
        self.Q = {key : 0 for key in self.Q}
        self.V = {key : 0 for key in self.V}
        self.Vtemp = {key : 0 for key in self.Vtemp}


    def update(self, state, action, Qold, delta):
        self.Q[(state, action)] = Qold + delta
    
    
    def evaluate(self, state, action):
        return self.Q[(state, action)]


    def update_V(self, state, delta):
        self.V[state] = delta

    def update_Vtemp(self, state, delta):
        self.Vtemp[state] = delta

    
    def evaluate_V(self, state):
        return self.V[state]


    def evaluate_Vtemp(self, state):
        return self.Vte[state]
    

    def update_V_from_Vtemp(self):
        # copy from Vtemp
        for state in self.mdp.states:
            self.V[state] = self.Vtemp[state]


    def update_V_from_Q(self):
        for state in self.mdp.states:
            actions = self.mdp.get_actions(state)
            self.V[state] =  max([self.Q[(state, action)] for action in actions])

  
    def display(self):

        # display values        
        print("-----------------------")
        for y in range(self.mdp.height-1, -1, -1):
            for x in range(self.mdp.width):
                if (x,y) in self.V:
                    print(f"{self.evaluate_V((x,y)): 0.2f}", end=' ')
                else:
                    print(f"{0.0: 0.2f}", end=' ')
            print("")       
        print("-----------------------")

        # policy extraction
        pi = self.extract_policy(self.mdp)
        print("-----------------------")
        for y in range(self.mdp.height-1, -1, -1):
            for x in range(self.mdp.width):
                if (x,y) in pi:
                    print(f"{pi[(x,y)]:<6}", end=' ')
                else:
                    print(f"{'None':<6}", end=' ')
            print("")       
        print("-----------------------")


class QTablePartial(QFunction):
    def __init__(self, mdp, default=0.0):
        self.mdp = mdp
        self.default = default
        # initialize Q-function and value function 
        self.Q = {}
        self.V = {}
        self.Vtemp = {}
        
        '''
        for state in (self.mdp.states):
            self.V[state] = default
            self.Vtemp[state] = default
            actions = self.mdp.get_actions()
            for action in actions:
                self.Q[(state, action)] = default
        '''

    def reset(self):
        self.Q = {key : 0 for key in self.Q}
        self.V = {key : 0 for key in self.V}
        self.Vtemp = {key : 0 for key in self.Vtemp}


    def update(self, state, action, Qold, delta):
        self.Q[(state, action)] = Qold + delta
    
    
    def evaluate(self, state, action):
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 0.0
        return self.Q[(state, action)]


    def update_V(self, state, delta):
        self.V[state] = delta

    def update_Vtemp(self, state, delta):
        self.Vtemp[state] = delta

    
    def evaluate_V(self, state):
        if state not in self.V:
            self.V[state] = 0.0
        return self.V[state]


    def evaluate_Vtemp(self, state):
        if state not in self.V_temp:
            self.V_temp[state] = 0.0
        return self.V_temp[state]
    

    def update_V_from_Vtemp(self):
        # copy from Vtemp
        for state in self.Vtemp:
            self.V[state] = self.Vtemp[state]


    def update_V_from_Q(self):
        for (state, action) in self.Q:
            if state in self.V:
                self.V[state] = max(self.Q[(state, action)],self.V[state])
            else:
                self.V[state] = self.Q[(state, action)]    
  
    def display(self):

        # display values        
        print("-----------------------")
        for y in range(self.mdp.height-1, -1, -1):
            for x in range(self.mdp.width):
                if (x,y) in self.V:
                    print(f"{self.evaluate_V((x,y)): 0.2f}", end=' ')
                else:
                    print(f"{0.0: 0.2f}", end=' ')
            print("")       
        print("-----------------------")

        # policy extraction
        pi = self.extract_policy(self.mdp)
        print("-----------------------")
        for y in range(self.mdp.height-1, -1, -1):
            for x in range(self.mdp.width):
                if (x,y) in pi:
                    print(f"{pi[(x,y)]:<6}", end=' ')
                else:
                    print(f"{'None':<6}", end=' ')
            print("")       
        print("-----------------------")




# value iteration loop
def value_iteration(grid_world_mdp, qtable, num_iters, theta=0.001):
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
                    Qnew += p * (grid_world_mdp.get_rewards(state, action, next_state) + grid_world_mdp.gamma * qtable.evaluate_V(next_state))
                qtable.update(state, action, 0.0, Qnew)
                Qsa.append(Qnew)
            Vnew = max(Qsa)    
            qtable.update_Vtemp(state, Vnew)  
            # update delta        
            d = max(d, abs(Vnew - qtable.evaluate_V(state)))
        
        # update the value function
        qtable.update_V_from_Vtemp()

        print("-----------------------")
        for y in range(grid_world_mdp.height-1, -1, -1):
            for x in range(grid_world_mdp.width):
                if (x,y) in qtable.V:
                    print(f"{qtable.evaluate_V((x,y)): 0.2f}", end=' ')
                else:
                    print(f"{0.0: 0.2f}", end=' ')
            print("")       
        print("-----------------------")


        # stop value iteration if delta falls below threshold    
        if d < theta:
            break    


# epsilon-greedy multi-arm bandit
class EpsGreedyBandit:

    def __init__(self, epsilon=0.1) -> None:
        self.epsilon = epsilon


    def select(self, state, actions, qfunction):    
        randnum = np.random.random()
        
        # exploration
        if randnum < self.epsilon:
            action = random.choices(actions, k=1)[0]
        
        # exploitation
        else:
            # argmax to find best action
            action, _ = qfunction.get_maxQ(state, actions)   

        if action == None:
            raise Exception("Error! Bandit action is None!")    

        return action
    

# upper confidence bounds (UCB) bandit
class UCBBandit:
    def __init__(self):
        self.total = 0
        # dictionary for recording number of times each action has been chosen
        self.times_selected = {}


    def select(self, state, actions, qfunction):
        
        # first, make sure each action has been executed once
        for action in actions:
            if action not in self.times_selected.keys():
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_actions = []
        max_value = float("-inf")
        for action in actions:
            value = qfunction.evaluate(state, action) + math.sqrt(2.0*math.log(self.total)/self.times_selected[action])
            if value > max_value:
                max_value = value
                max_actions = [action]
            elif value == max_value:
                max_actions.append(action)

        # if multiple actions with max value, pick one randomly
        selected_action = random.choice(max_actions)
        self.times_selected[selected_action] = self.times_selected[selected_action] + 1
        self.total += 1
        
        return selected_action          
            

# Q-learner class
class QLearner:
    def __init__(self, mdp, qfunction, alpha=0.1, epsilon=0.1) :
        self.mdp = mdp
        self.qfunction = qfunction
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
            # argmax to find best action
            action, _ = self.qfunction.get_maxQ(state, actions)   

        if action == None:
            raise Exception("Error! Bandit action is None!")    

        return action


    # Q-learning update
    def get_delta_Q(self, reward, Qold, state, next_state, next_action):
        # get estimated value for next state
        actions = self.mdp.get_actions(next_state)
        _, Vsprime = self.qfunction.get_maxQ(next_state, actions) 
        delta = reward + self.mdp.gamma * Vsprime - Qold
        return self.alpha * delta  


    # SARSA update
    def get_delta_SARSA(self, reward, Qold, state, next_state, next_action):
        # get estimated value for next state
        Vsprime = self.qfunction.evaluate(next_state, next_action) 
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
                Qold = self.qfunction.evaluate(state, action)
                if not SARSA:
                    delta = self.get_delta_Q(reward, Qold, state, next_state, next_action)
                else:
                    delta = self.get_delta_SARSA(reward, Qold, state, next_state, next_action)
                self.qfunction.update(state,action,Qold,delta)
                state = next_state
                action = next_action
                
                # accumulate discounted reward for the step
                accumulated_discounted_reward += reward * (self.mdp.gamma**episode_len)
                episode_len += 1

            # save total accumulated reward after episode is over 
            episode_rewards.append(accumulated_discounted_reward)
            print(f"Episode# {i}, length: {episode_len}, accumulated reward: {accumulated_discounted_reward}")
   
            # update the value function
            self.qfunction.update_V_from_Q()

            print("-----------------------")
            for y in range(self.mdp.height-1, -1, -1):
                for x in range(self.mdp.width):
                    if (x,y) in self.qfunction.V:
                        print(f"{self.qfunction.evaluate_V((x,y)): 0.2f}", end=' ')
                    else:
                        print(f"{0.0: 0.2f}", end=' ')
                print("")       
            print("-----------------------")

        return episode_rewards


def moving_average(x, window_size):
    smoothed_x = []
    for i in range(len(x)):
        lo = max(0, i-window_size+1)
        windowed = x[lo:i+1]
        avg = sum(windowed) / len(windowed)
        smoothed_x.append(avg)
    return smoothed_x

def plot_rewards(rewards, smoothing=True, window_size=10):
    fig = plt.figure(figsize=(5, 5))
    if smoothing:
        plt.plot(moving_average(rewards, window_size))
    else:
        plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward")
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.show()
