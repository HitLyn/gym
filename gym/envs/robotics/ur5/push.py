import os
from gym import utils
from gym.envs.robotics import ur5_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('ur5', 'push.xml')


class UR5PushEnv(ur5_env.UR5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'ur5_shoulder_pan_joint': 1.52,
            'ur5_shoulder_lift_joint': -0.89,
            'ur5_elbow_joint': 1.38,
            'ur5_wrist_1_joint': -2.05,
            'ur5_wrist_2_joint': -1.54,
            'ur5_wrist_3_joint': 0.0,
            's_model_finger_1_joint_1': 0.73,
            's_model_finger_1_joint_2': 0.13,
            's_model_finger_1_joint_3': -0.64,
            's_model_finger_2_joint_1': 0.88,
            's_model_finger_2_joint_2': 0.,
            's_model_finger_2_joint_3': -0.64,
            's_model_finger_middle_joint_1': 0.86,
            's_model_finger_middle_joint_2': 0.,
            's_model_finger_middle_joint_3': 0.,
            's_model_palm_finger_1_joint': 0.,
            's_model_palm_finger_2_joint': 0.,
            'object0:joint': [0., 0., 0., 1., 0., 0., 0.],
        }

        # initial_qpos = {
        #     'ur5_shoulder_pan_joint': 0.,
        #     'ur5_shoulder_lift_joint': -0.,
        #     'ur5_elbow_joint': 0.,
        #     'ur5_wrist_1_joint': -0.,
        #     'ur5_wrist_2_joint': -0.,
        #     'ur5_wrist_3_joint': 0.0,
        #     's_model_finger_1_joint_1': 0.73,
        #     's_model_finger_1_joint_2': 0.13,
        #     's_model_finger_1_joint_3': -0.64,
        #     's_model_finger_2_joint_1': 0.88,
        #     's_model_finger_2_joint_2': 0.,
        #     's_model_finger_2_joint_3': -0.64,
        #     's_model_finger_middle_joint_1': 0.86,
        #     's_model_finger_middle_joint_2': 0.,
        #     's_model_finger_middle_joint_3': 0.,
        #     's_model_palm_finger_1_joint': 0.,
        #     's_model_palm_finger_2_joint': 0.,
        #     'object0:joint': [0., 0., 0., 1., 0., 0., 0.],
        # }
        # print(initial_qpos)
        ur5_env.UR5Env.__init__(
            self, MODEL_XML_PATH, n_substeps=20,
            pusher_extra_height=0.0, obj_range=0.15, target_range=0.15,
            block_gripper = True, distance_threshold=0.05, initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
