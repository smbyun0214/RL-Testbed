import cv2
import numpy as np

from gym import Wrapper
from gym import Env

class MyWrapper(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
    
    def reset(self):
        self.env.reset()
        obs = self.env.render(mode="rgb_array")
        new_obs = self.preprocessing(obs)
        return new_obs
    
    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.render(mode="rgb_array")
        new_obs = self.preprocessing(obs)
        return new_obs, reward, done, info

    def preprocessing(self, obs):
        new_obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        new_obs = cv2.resize(new_obs, (320, 320))
        new_obs = cv2.erode(new_obs, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        new_obs = cv2.resize(new_obs, (80, 80))
        _, new_obs = cv2.threshold(new_obs, 220, 255, cv2.THRESH_BINARY_INV)
        return np.expand_dims(new_obs, axis=-1)
