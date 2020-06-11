import os
import numpy as np
from gym import utils
from gym.envs.robotics import rotations, fetch_env, fetch_generate_env
from gym.envs.robotics import utils as robotics_utils


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_generate.xml')

class FetchPushGenerateEnv(fetch_generate_env.FetchGenerateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, -0.5, 0., 0., 0.868],
        }


        fetch_generate_env.FetchGenerateEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20,
            initial_qpos=initial_qpos, image_path = '/home/lyn/generated_images')
        utils.EzPickle.__init__(self)
