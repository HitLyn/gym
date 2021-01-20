import numpy as np
import matplotlib.pyplot as plt
import copy

import os

from gym.envs.robotics import robot_env, utils, rotations
from gym.envs.robotics.rotations import quat_from_angle_and_axis

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class FetchGenerateEnv(robot_env.RobotEnv):
    """
        Superclass for image generate environments, a simple version of the gym env without any meaningful reward or observations
        functions related are defined only to satisfy the overall structure requirentments.
    """

    def __init__(self, model_path, n_substeps, initial_qpos, image_path, has_object = True):
        self.image_path = image_path
        self.object_target_range, self.robot_target_range = self._get_target_range()
        self.has_object = has_object
        # self.object = self.reset_object()

        super(FetchGenerateEnv, self).__init__(model_path = model_path, n_substeps = n_substeps, n_actions = 3, initial_qpos = initial_qpos)
    def compute_reward(self, achieved_goal, goal, info):
        return 0

    def _set_action(self, action):
        assert action.shape == (3,)
        action = action.copy() # make sure action keep the same outside of this scope
        pos_ctrl = 0.05*action[:3]
        rot_ctrl = [1., 0., 1., 0.]
        action = np.concatenate([pos_ctrl, rot_ctrl])
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = self.sim.data.get_joint_qpos('object0:joint')

        obs = grip_pos

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

        # turn off the overlay
        # self.viewer._hide_overlay = True

    def _render_callback(self):

        # goal render
        goal = self.goal.copy()
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        # object render
        object = self.object.copy()
        self.sim.data.set_joint_qpos('object0:joint', object)
        self.sim.data.set_joint_qvel('object0:joint', np.zeros(6))

        self.sim.forward()

        # robot render


    def render(self, mode='rgb_array'):
        # print("viewer mode: {}".format(mode))
        return super(FetchGenerateEnv, self).render(mode)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        # if self.has_object:
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #         object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #     object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True


    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.31]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
            self.reset_object()

    def _is_success(self, achieved_goal, desired_goal):
        # d = goal_distance(achieved_goal, desired_goal)
        # return (d < self.distance_threshold).astype(np.float32)
        return 0

    def _sample_goal(self):
        x_min, x_max, y_min, y_max = self.object_target_range
        new_goal = np.zeros(7)
        pos_x = np.random.uniform(x_min, x_max)
        pos_y = np.random.uniform(y_min, y_max)
        pos_z = self.height_offset
        # angle
        angle = np.random.uniform(-np.pi, np.pi)
        axis = np.array([0., 0., 1.])
        angle_quat = quat_from_angle_and_axis(angle, axis)
        angle_quat /= np.linalg.norm(angle)

        new_goal[:3] = [pos_x, pos_y, pos_z]
        new_goal[3:] = angle_quat
        self.goal = new_goal.copy()
        return new_goal.copy()

    def save_image(self, option, idx):
        # option (str): 'anchor', 'goal', 'object', 'robot'
        image = self.render()

        # mkdir or get access to the path
        if not os.path.exists(os.path.join(self.image_path, option)):
            os.makedirs(os.path.join(self.image_path, option))
        # else:
            # print('save image to {}'.format(os.path.join(self.image_path, option)))

        file_name = "{}/{:06d}.png".format(os.path.join(self.image_path, option), idx)
        plt.imsave(file_name, image)

    def reset_goal(self):
        x_min, x_max, y_min, y_max = self.object_target_range
        new_goal = np.zeros(7)
        pos_x = np.random.uniform(x_min, x_max)
        pos_y = np.random.uniform(y_min, y_max)
        pos_z = self.height_offset
        # angle
        angle = np.random.uniform(-np.pi, np.pi)
        axis = np.array([0., 0., 1.])
        angle_quat = quat_from_angle_and_axis(angle, axis)
        angle_quat /= np.linalg.norm(angle)

        new_goal[:3] = [pos_x, pos_y, pos_z]
        new_goal[3:] = angle_quat
        self.goal = new_goal.copy()

        # goal render
        # goal = self.goal.copy()
        self.sim.data.set_joint_qpos('target:joint', self.goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))
        return new_goal

    def reset_anchor(self):
        self.reset_object()
        self.reset_robot()
        self.reset_goal()


    def reset_object(self):
        x_min, x_max, y_min, y_max = self.object_target_range
        new_goal = np.zeros(7)
        pos_x = np.random.uniform(x_min, x_max)
        pos_y = np.random.uniform(y_min, y_max)
        pos_z = self.height_offset
        # angle
        angle = np.random.uniform(-np.pi, np.pi)
        axis = np.array([0., 0., 1.])
        angle_quat = quat_from_angle_and_axis(angle, axis)
        angle_quat /= np.linalg.norm(angle)

        new_goal[:3] = [pos_x, pos_y, pos_z]
        new_goal[3:] = angle_quat
        self.object = new_goal.copy()

        # object render
        # object = self.object.copy()
        self.sim.data.set_joint_qpos('object0:joint', self.object)
        self.sim.data.set_joint_qvel('object0:joint', np.zeros(6))
        return new_goal

    def reset_robot(self):
        x_min, x_max, y_min, y_max = self.robot_target_range
        pos_x = np.random.uniform(x_min, x_max)
        pos_y = np.random.uniform(y_min, y_max)
        pos_z = self.sim.data.get_mocap_pos('robot0:mocap')[2]
        gripper_target = np.array([pos_x, pos_y, pos_z])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def get_current_state(self):
        """
            return robot_state, object_state, goal_state
        """
        # get robot_state
        all_state = self.sim.get_state()
        goal_state = self.goal.copy()
        object_state = self.sim.data.get_joint_qpos('object0:joint').copy()

        return {'all_state': all_state, 'goal_state': goal_state, 'object_state': object_state}

    def recover_state(self, state):
        # state: {robot_state, goal_state, object_state}
        all_state = state['all_state']
        object_state = state['object_state']
        goal_state = state['goal_state']

        self.goal = goal_state.copy()
        self.object = object_state.copy()

        self.sim.set_state(all_state) # including robot, goal and target


    def _get_target_range(self):
        object_target_range = [1.0, 1.55, 0.45, 0.85]
        robot_target_range = [1.1, 1.45, 0.5, 0.8]

        return object_target_range, robot_target_range
