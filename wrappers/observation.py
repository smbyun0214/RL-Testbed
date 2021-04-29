import gym
import numpy as np
from PIL import Image


class WarpFrame(gym.ObservationWrapper):
    """Nature 논문에서 쓰인 84x84 frame으로 전처리 한다."""

    def __init__(self, env, width=84, height=84, grayscale=True):
        super(WarpFrame, self).__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        original_space = self.observation_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )

    def observation(self, obs):
        frame = obs

        image = Image.fromarray(obs)
        
        # cropping an 84 × 84 region of the image that roughly captures the playing area
        (left, upper, right, lower) = (0, 17, 160, 17+177)
        image = image.crop(box=(left, upper, right, lower))

        # converting their RGB representation to grayscale
        if self._grayscale:
            image = image.convert("L")

        # down-sampling it to a 110×84 image
        image = image.resize(size=(self._width, self._height))

        frame = np.array(image)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        obs = frame
        return obs
