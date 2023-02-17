
import torch 
import copy
from collections import deque
import random
import numpy as np


REPLAY_MEM_SIZE=10_000
BATCH_SIZE=32
MIN_REPLAY_SIZE=500
DISCOUNT=1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPDATE_TARGET_EVERY=50

class qAgent():

    def __init__(self, state_space, action_space) -> None:


        
        #initliaze online model
        self.model = DriveNet(state_space,action_space).to(DEVICE).double()
        

        #initliaze target model
        self.target_model=DriveNet(state_space,action_space).to(DEVICE).double()
        self.target_model.load_state_dict(self.model.state_dict())

        self.memory=deque(maxlen=REPLAY_MEM_SIZE)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
        self.loss = torch.nn.SmoothL1Loss()

        self.target_update_counter = 0





    def update_memory(self,state,action_ind,next_state,reward,done):

        state = torch.tensor(state, device=DEVICE).type(torch.DoubleTensor)
        next_state = torch.tensor(next_state, device=DEVICE).type(torch.DoubleTensor)
        action_ind  = torch.tensor(action_ind, device=DEVICE)
        reward = torch.tensor(reward, device=DEVICE)
        done = torch.tensor(done, device=DEVICE)

        self.memory.append((state, action_ind, next_state , reward, done,))


    def get_qs(self, state):
        return self.model(state)

    def train(self,terminal_state,step):
        if len(self.memory) < MIN_REPLAY_SIZE:
            return

        minibatch=random.sample(self.memory,BATCH_SIZE)
        states, actions, next_states, reward, done = map(torch.stack, zip(*minibatch))


        current_qs_list=self.model(states.to(DEVICE))

        future_qs_list=self.target_model(next_states.to(DEVICE))

        X=[]
        Y=[]

        for index, (current_state,action,nextstate,reward,done) in enumerate(minibatch):
            if not done:
                max_future_q = torch.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index] 
            current_qs[action]=new_q

            X.append(self.model(current_state))
            Y.append(current_qs)

        Y=torch.stack(Y)
        X=torch.stack(X)
        loss = self.loss(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1


        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0


class DriveNet(torch.nn.Module):
    # Create a nn architecture for 
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.learn = torch.nn.Sequential(
        torch.nn.Linear(input_dim,124),
        torch.nn.ReLU(),
        torch.nn.Linear(124,512),
        torch.nn.ReLU(),
        torch.nn.Linear(512,512),
        torch.nn.ReLU(),
        torch.nn.Linear(512,output_dim)   
        )

    def forward(self, input):
        #input = input.to(DEVICE)
        return self.learn(input)