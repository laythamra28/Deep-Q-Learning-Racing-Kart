# run pip install -r reqs.txt to install dependencies
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

#Pystk libraries
import pystk

SteerAction = np.array([-1,0,1])
SteerAction = np.reshape(SteerAction,(3,1))



## Intializing Enviornment 
class RaceEnviroment():
    def __init__(self, screen_width=128, screen_height=96) -> None:
        RaceEnviroment._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.race=None
        self.action=pystk.Action()

    def _point_on_track(self,distance,track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """

        
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    def _to_image(self,x, proj, view):
        # input x is return of _point_on_track, return of to_image is aim point
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)


    
    def start_race(self,track):
        config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        self.race = pystk.Race(config)
        self.race.start()
        self.last_rescue_t = 0 # initalize last rescue
        self.t = 0 # time of starting the race
        self.state = pystk.WorldState()
        self.track = pystk.Track()
        self.lastAimPoint=np.array([0])
    

    def step(self,track,action_ind):
        #this function takes an action and takes a step using that action then calculates aimpoint of next state
        
        
        #if there is a race and its on the wrong track delete the race and start a new race on the right track

        if self.race is not None and self.race.config.track!= track: 
            self.start_race(track)
  
        #If no race has been done yet then start a race
        if self.race is None:
            self.start_race(track)
            
        
        # initalize reward for each step as 0
        reward = 0 

        
        #update state and track 
        self.state.update()
        self.track.update()

        #find current velocity
        kart = self.state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)


        #input action given for steer
        self.action.steer=SteerAction[action_ind]

        #Hardcoded actions
        target_velocity = 9

        if (current_vel < target_velocity):
            self.action.acceleration = 0.9    
        else:
            self.action.acceleration = 0.1


        #take a step in the enviroment given action
        self.race.step(self.action)


        #update state and track 
        self.state.update()
        self.track.update()

       
   

        #See if done
        done=False
        done = self.state.players[0].kart.race_result
        location = self.state.players[0].kart. distance_down_track
        

        self.t = self.t+1

        # give -1 reward for each frame spent on track
        #reward = reward-1 
        

        #calculate variables needed for aim pointer
        proj = np.array(self.state.players[0].camera.projection).T
        view = np.array(self.state.players[0].camera.view).T
    
        #calculate aim pointer
        aim_point_world = self._point_on_track(kart.distance_down_track+15, self.track)
        aim_point_image = np.round(self._to_image(aim_point_world, proj, view),1)
        

        #See if kart needs to be rescued
        self.action.rescue = False
        
        if current_vel < 1.0 and self.t - self.last_rescue_t > 30:
                self.last_rescue_t = self.t
                self.action.rescue = True
                reward = reward - 30 #give action -1000 reward if needed to be resecuted


        #if done give reward
        if done == True:
            reward+=500

        #if last aimpoint near this aimpoint give reward
        if abs(self.lastAimPoint) - abs(aim_point_image[0]) < .15 and self.action.rescue == False:
            reward += .05
            
        else:
            reward += -.02


        self.lastAimPoint = aim_point_image[0]

        if done == True:
            self.reset(track)
            
        return aim_point_image, reward, done


    def reset(self,track):
        self.race.stop()
        del self.race
        #pystk.clean()
        #self.__init__()
        self.start_race(track)
        
        
        

