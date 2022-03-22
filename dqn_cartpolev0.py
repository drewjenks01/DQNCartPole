import tensorflow as tf
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense
from keras.optimizers import adam_v2
import numpy as np
from collections import deque
import random
import gym
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from collections import deque
import copy
from time import time


class DQN:

    #intialize DQN class
    def __init__(self):

        #initialize environment
        self.env = gym.make('CartPole-v0')
        self.env.reset(seed=1)

        #define the number of actions and states in environment
        self.numActions = self.env.action_space.n
        self.numStateVar = self.env.observation_space.shape[0]

        #define epsilon for epsilon-greedy policy
        self.epsilon = 0.80

        #epsilon decay rate -- for exploration -> exploitation
        self.epsilon_decay = 0.95

        #lowest epsilon value (keep a little to allow small percentage of random action even later in training)
        self.epsilon_cutoff = 0.01

        #define learning rate for model
        self.alpha = 0.0005

        #discount factor for Bellman eq.
        self.gamma = 0.80

        #number of episodes to perform during training
        self.num_eps = 100

        #file to load saved weights (option)
        self.weight_file = None

        #training batch size --> potential improvement: increase batch size over time
        self.batchSize = 64

        #define memory storage
        self.memory = deque(maxlen=100000)

        #how much memory allocation for initial random actions before training
        self.startMemSize = 100

        #number of steps to perform per episode
        self.num_steps = 200

        #number of epochs during model.fit
        self.numEpoch = 1

        #define out dqn model
        self.Q = None

        #define our target dqn model --> updates occasionally
        self.QHat = None

        #how often to update target net (# steps)
        self.Qhat_update = 10

        #hold training losses
        self.training_losses = np.empty(0)

        #hold training rewards
        self.training_rewards = np.empty(0)

        #how often to print out training results
        self.interval=5


    #function to create DQN model
    def build_model(self):
        """
        Function to build a DQN model.Uses ReLu activation, Adam optimizer with mean squared error loss 
        """

        #initialize sequential model
        self.Q=Sequential()


        self.Q.add(Dense(128,input_dim=self.numStateVar,activation='relu',kernel_initializer='he_uniform'))

        #add first dense layer, relu activation
        self.Q.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))

        #add second layer, relu activation
        self.Q.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))

        #add last linear output layer
        self.Q.add(Dense(2,activation='linear',kernel_initializer='he_uniform'))

        #Adam optimizer
        opt=adam_v2.Adam(learning_rate=self.alpha)

        #compile model
        self.Q.compile(loss='mse', optimizer=opt)

        #print summary of model
        self.Q.summary()

        #define our target model
        self.QHat = clone_model(self.Q)
        self.QHat.set_weights(self.Q.get_weights())


    #function to run one game
    def one_game(self,eps,render,train):
        """
        Function to run one episode of the game.

        Args:
            eps (float): epsilon val
            render (bool): render environment?
            train (bool): training?

        Returns:
            bool: score from playing the game
        """
        #keep track of total reward
        score=0

        #reset env
        state=self.env.reset(seed=1)

        #iterate # of steps
        for i in range(self.num_steps):
            #choose an action based on policy and state
            action=self.policy(eps,state)

            #step env and store components
            new_state,reward,done,_=self.env.step(action)

            #add components to memory
            self.memory.append((state,action,reward,new_state,done))

            #check if render
            if render:
                self.env.render()

            #check if training model
            if train:

                #call train() for model
                self.train_model()

                #update target model occasionally
                if i%self.Qhat_update==0:
                    self.QHat.set_weights(self.Q.get_weights())

            
            #set state equal to new state
            state=new_state

            #add score to total
            score+=reward

            #check if terminal state
            if done :
                break

        return score


    #function to train model
    def train_model(self):
        """
        Function to train model.
        """

        #create a training set from random sample from memory
        train_batch=random.sample(self.memory,self.batchSize)

        #define our input and output matrix shapes
        X=np.empty(shape=(0,self.numStateVar),dtype=float)
        Y=np.empty(shape=(0,self.numActions),dtype=float)

        #seperate training batch into components
        states=np.array([data[0] for data in train_batch])
        new_states=np.array([data[3] for data in train_batch])

        #collect all Q and target cals
        Qs=self.Q.predict(states)
        targets=self.QHat.predict(new_states)

        #iterate through training batch
        for i in range(self.batchSize):
            state=np.reshape(train_batch[i][0],(1,self.numStateVar))
            action=train_batch[i][1]
            reward=train_batch[i][2]
            done=train_batch[i][4]

            curr_Q=copy.copy(Qs[i])

            #if not terminal state, update using Bellman
            if not done:
                max_target=np.max(targets[i])
                #print(max_target,curr_Q,curr_Q[action])
                curr_Q[action]=reward+self.gamma*max_target

            #if terminal state, Q=reward
            elif done:
                curr_Q[action]=reward

            #reshape Qs
            curr_Q=np.reshape(curr_Q,(1,self.numActions))

            #add sample to training set
            X=np.append(X,state,axis=0)
            Y=np.append(Y,curr_Q,axis=0)

        #fit the model to training set --> this implements SGD for you
        res=self.Q.fit(X,Y,batch_size=self.batchSize,epochs=self.numEpoch,verbose=0)

        #append loss
        self.training_losses=np.append(self.training_losses,res.history['loss'][-1])

  
    #policy function
    def policy(self, eps, state):
        """
        Policy function to decide what action to take.

        Args:
            eps (bool): epsilon value
            state (int): state that was observed in environment

        Returns:
            int: action to take
        """
        # exploration --> higher eps means higher chance of taking random action
        if np.random.random() < eps:
            action = self.env.action_space.sample()

        # exploitation --> when our model chooses the best action to take
        else:
            action = np.argmax(self.Q.predict(np.reshape(state, (1, self.numStateVar)))[0])
        return action


    #function to load saved model
    def upload_model(self,mod):
        """
        Function that uploads a saved model.

        Args:
            mod (DQN model): saved DQN model
        """
        #set saved model
        self.Q=mod


    #function to run training
    def training(self):
        """
        Function to handle all of training
        """
        #define the epsilon val
        eps=self.epsilon
        
        #populate memory with random actions
        while len(self.memory)<self.startMemSize:
            self.one_game(eps=1,render=False,train=False)

        #store train times
        times=[]

        #train model
        for i in range(self.num_eps):
            #start time before runnning a game
            start=time()

            #run one episode of training
            reward=self.one_game(eps,render=False,train=True)

            #end time after running a game
            end=time()

            #time it took to train on one epsiode
            times.append(end-start)

            #store reward from one epsiode
            self.training_rewards=np.append(self.training_rewards,reward)

             
            #display message during training to track progress
            if i % self.interval == 0 and i > 0:

                #increase batch size over time
                self.batchSize+=2

                #save weights
                self.Q.save('/Users/drewj/Documents/PulkitUROP/nano quiz - cartpullv0/saved/saved_model.h5')

                print("\nepisode {}: loss-->{:.2f}+/-{:.2f}, reward-->{:.2f}+/-{:.2f}, time-->{:.2f}sec".format(i, 
                    np.average(self.training_losses[-self.interval:-1]),
                    np.std(self.training_losses[-self.interval:-1]),  
                    np.average(self.training_rewards[-self.interval:-1]),
                    np.std(self.training_rewards[-self.interval:-1]),
                    np.average(np.array(times))))
                times=[]

            #update epsilon value w/ decay
            if eps >self.epsilon_cutoff:
                eps*=self.epsilon_decay


    #function to evaluate testing
    def eval_model(self,numRuns):
        """
        Function to evalute model (test) after training

        Args:
            numRuns (int): how many episodes to run

        Returns:
            str: display message containing avg rewards over all episodes
        """
        rewards = np.empty(0)

        #iterate through number of episodes defined
        for i in range(numRuns):
            #run a game...eps=0 for no exploration
            rew=self.one_game(0, render=True, train=False)

            #store reward from game
            rewards = np.append(rewards,rew )
            print('test '+str(i)+', score: ',rew)
        return 'Avg reward for ' +str(numRuns)+':',sum(rewards)/len(rewards)



#run the file
if __name__ == '__main__':




    #initialize dqn
    dqn=DQN()

    #build dqn
  #  dqn.build_model()

    #train dqn
   # dqn.training()

    #test dqn
    dqn.upload_model(load_model('/Users/drewj/Documents/PulkitUROP/nano quiz - cartpullv0/saved/model_success.h5'))

    res=dqn.eval_model(10)


    print(res)




