import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5Env(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, pusher_extra_height, obj_range, target_range,
        block_gripper, distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            pusher_extra_height (float): additional height above the table when positioning the pusher
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.pusher_extra_height = pusher_extra_height
        self.block_gripper = block_gripper
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(UR5Env, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=2,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('s_model_palm_finger_1_joint', 0.)
            self.sim.data.set_joint_qpos('s_model_finger_1_joint_1', 1.)
            self.sim.data.set_joint_qpos('s_model_finger_1_joint_2', 1.)
            self.sim.data.set_joint_qpos('s_model_finger_1_joint_3', -1.)
            self.sim.data.set_joint_qpos('s_model_palm_finger_2_joint', 0.)
            self.sim.data.set_joint_qpos('s_model_finger_2_joint_1', 1.)
            self.sim.data.set_joint_qpos('s_model_finger_2_joint_2', 1.)
            self.sim.data.set_joint_qpos('s_model_finger_2_joint_3', -1.)
            self.sim.data.set_joint_qpos('s_model_finger_middle_joint_1', 1.)
            self.sim.data.set_joint_qpos('s_model_finger_middle_joint_2', 1.)
            self.sim.data.set_joint_qpos('s_model_finger_middle_joint_3', -1.)
            self.sim.forward()

    def _set_action(self, action):
        # maybe add gripper contrl here
        assert action.shape == (2,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        # utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        pusher_pos = self.sim.data.get_site_xpos('pusher')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        pusher_velp = self.sim.data.get_site_xvelp('pusher') * dt
        robot_qpos, robot_qvel = utils.ur5_get_obs(self.sim)

        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # pusher state
        object_rel_pos = object_pos - pusher_pos
        object_velp -= pusher_velp

        # achieved_goal = np.squeeze(object_pos.copy())
        achieved_goal = np.concatenate(object_pose.ravel(), object_rot.ravel())
        obs = np.concatenate([
            pusher_pos, object_pos.ravel(), object_rot.ravel(),
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('s_model_palm')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        object_xpos = self.initial_pusher_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_pusher_xpos[:2]) < 0.1:
            object_xpos = self.initial_pusher_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_pusher_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        pusher_target = np.array([0.89, 0.74, 0.93])
        pusher_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('ur5_mocap', gripper_target)
        self.sim.data.set_mocap_quat('ur5_mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_pusher_xpos = self.sim.data.get_site_xpos('pusher').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(UR5Env, self).render(mode, width, height)
