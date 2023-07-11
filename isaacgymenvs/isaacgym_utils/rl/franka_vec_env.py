import numpy as np
import quaternion

from isaacgym import gymapi
import gym
from isaacgym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from isaacgym_utils.math_utils import np_to_vec3, rpy_to_quat, transform_to_np
from isaacgym_utils.ctrl_utils import ForcePositionController, MovingMedianFilter
# import torch

from gym.spaces import Box

from .vec_env import GymVecEnv


class GymFrankaVecEnv(GymVecEnv):

    _ACTUATION_MODE_MAP = {
                            'vic': 'attractors',
                            'hfpc': 'torques',
                            'hfpc_cartesian_gains': 'torques',
                            'joints': 'joints'
                        }

    def _setup_single_env_gen(self, cfg):
        self._table = GymBoxAsset(self._scene, **cfg['table']['dims'],
                            shape_props=cfg['table']['shape_props'],
                            asset_options=cfg['table']['asset_options']
                            )
        self._actuation_mode = self._ACTUATION_MODE_MAP[cfg['franka']['action']['mode']]
        franka = GymFranka(cfg['franka'], self._scene, actuation_mode=self._actuation_mode)
        self._frankas = [franka] * self.n_envs

        table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
        self._franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))

        self._franka_name = 'franka'
        self._table_name = 'table'

        def setup(scene, _):
            scene.add_asset(self._table_name, self._table, table_transform)
            scene.add_asset(self._franka_name, franka, self._franka_transform, collision_filter=1) # avoid self-collisions
        return setup

    def _init_action_space(self, cfg):
        action_cfg = cfg['franka']['action'][cfg['franka']['action']['mode']]
        self._action_mode = cfg['franka']['action']['mode']

        if self._action_mode == 'vic':
            limits_low = np.array(
                [-action_cfg['max_tra_delta']] * 3 + \
                [-np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                [action_cfg['min_stiffness']])
            limits_high = np.array(
                [action_cfg['max_tra_delta']] * 3 + \
                [np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                [action_cfg['max_stiffness']])

            self._init_ee_transforms = [
                self._frankas[env_idx].get_desired_ee_transform(env_idx, self._franka_name)
                for env_idx in range(self.n_envs)
            ]
        elif self._action_mode == 'hfpc':
            self._force_filter = MovingMedianFilter(6, 3)
            self._fp_ctrlr = ForcePositionController(
                np.zeros(6), np.zeros(6), np.ones(6), 7)
            limits_low = np.array(
                        [-action_cfg['max_tra_delta']] * 3 + \
                        [-np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [-np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [0] * 3 + \
                        [action_cfg['min_pos_kp'], action_cfg['min_force_kp']])
            limits_high = np.array(
                        [action_cfg['max_tra_delta']] * 3 + \
                        [np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [1] * 3 + \
                        [action_cfg['max_pos_kp'], action_cfg['max_force_kp']])
        elif self._action_mode == 'hfpc_cartesian_gains':
            self._force_filter = MovingMedianFilter(6, 3)
            self._fp_ctrlr = ForcePositionController(
                np.zeros(6), np.zeros(6), np.ones(6), 7,
                use_joint_gains_for_position_ctrl=False,
                use_joint_gains_for_force_ctrl=False
            )

            limits_low = np.array(
                        [-action_cfg['max_tra_delta']] * 3 + \
                        [-np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [-np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [0] * 3 + \
                        [action_cfg['min_pos_kp'], action_cfg['min_force_kp']])
            limits_high = np.array(
                        [action_cfg['max_tra_delta']] * 3 + \
                        [np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [1] * 3 + \
                        [action_cfg['max_pos_kp'], action_cfg['max_force_kp']])

        elif self._action_mode == 'joints':
            max_rot_delta = np.deg2rad(action_cfg['max_rot_delta'])
            limits_high = np.array([max_rot_delta] * 7)
            limits_low = -limits_high
        else:
            raise ValueError('Unknown action mode!')

        # gripper action
        limits_low = np.concatenate([limits_low, [0]])
        limits_high = np.concatenate([limits_high, [1]])

        action_space = Box(limits_low, limits_high, dtype=np.float32)
        return action_space

    def _init_obs_space(self, cfg):
        '''
        Observations contains:

        joint angles - 7
        gripper width (0 to 0.08) - 1
        ee position - 3
        ee quat - 4
        ee contact forces - 3
        '''
        limits_low = np.array(
            self._frankas[0].joint_limits_lower.tolist()[:-2] + \
            [0] + \
            [-10] * 3 + \
            [-1] * 4 + \
            [-1e-5] * 3
        )
        limits_high = np.array(
            self._frankas[0].joint_limits_upper.tolist()[:-2] + \
            [0.08] + \
            [10] * 3 + \
            [1] * 4 + \
            [1e-5] * 3
        )
        obs_space = Box(limits_low, limits_high, dtype=np.float32)
        self.progress_buf = np.zeros(self.num_envs)
        return obs_space

    def _apply_actions(self, all_actions):
        for env_idx in self._scene.env_idxs:
            action = all_actions[env_idx]
            arm_action = action[:-1]

            if self._action_mode == 'vic':
                delta_tra = arm_action[:3]
                delta_rpy = arm_action[3:6]
                stiffness = arm_action[6]

                self._frankas[env_idx].set_attractor_props(env_idx, self._franka_name,
                {
                    'stiffness': stiffness,
                    'damping': 2 * np.sqrt(stiffness)
                })

                delta_transform = gymapi.Transform(
                    p=np_to_vec3(delta_tra),
                    r=rpy_to_quat(delta_rpy),
                )
                self._frankas[env_idx].set_delta_ee_transform(env_idx, self._franka_name, delta_transform)
            elif self._action_mode == 'hfpc' or self._action_mode == 'hfpc_cartesian_gains':
                xa_tf = self._frankas[env_idx].get_ee_transform(env_idx, self._franka_name)
                xa = transform_to_np(xa_tf, format='wxyz')

                fa = -np.concatenate([self._frankas[env_idx].get_ee_ct_forces(env_idx, self._franka_name),
                                      [0,0,0]], axis=0)
                fa = self._force_filter.step(fa)
                J = self._frankas[env_idx].get_jacobian(env_idx, self._franka_name)

                # The last two points are finger joints.
                qdot = self._frankas[env_idx].get_joints_velocity(env_idx, self._franka_name)[:7]
                xdot = np.matmul(J, qdot)

                xd_position = xa[:3] + arm_action[:3]
                xd_orient_rpy = arm_action[3:6]
                xd_orient_quat = quaternion.from_euler_angles(xd_orient_rpy)
                xd_orient = quaternion.as_float_array(xd_orient_quat)
                xd = np.concatenate([xd_position, xd_orient])

                fd = np.concatenate([fa[:3] + arm_action[6:9], [0, 0, 0]])
                S = np.concatenate([arm_action[9:12], [1, 1, 1]])

                pos_kp, force_kp = arm_action[12:14]
                pos_kd = 2 * np.sqrt(pos_kp)
                force_ki = 0.01 * force_kp
                self._fp_ctrlr.set_ctrls(force_kp, force_ki, pos_kp, pos_kd)
                self._fp_ctrlr.set_targets(xd=xd, fd=fd, S=S)

                tau = self._fp_ctrlr.step(xa, xdot, fa, J, qdot)
                self._frankas[env_idx].apply_torque(env_idx, self._franka_name, tau)
            elif self._action_mode == 'joints':
                delta_joints = np.concatenate([arm_action, [0, 0]]) # add dummy gripper joint cmds
                self._frankas[env_idx].apply_delta_joint_targets(env_idx, self._franka_name, delta_joints)
            else:
                raise ValueError(f"Invalid action mode: {self._action_mode}")

            gripper_action = action[-1]
            gripper_width = np.clip(gripper_action, 0, 0.04)
            self._frankas[env_idx].set_gripper_width_target(env_idx, self._franka_name, gripper_width)

    def _compute_obs(self, all_actions):
        all_obs = np.zeros((self.n_envs, 18))
        self.l_finger_pos = np.zeros((self.n_envs, 3))
        self.r_finger_pos = np.zeros((self.n_envs, 3))

        for env_idx in self._scene.env_idxs:
            all_joints = self._frankas[env_idx].get_joints(env_idx, self._franka_name)
            ee_transform = self._frankas[env_idx].get_ee_transform(env_idx, self._franka_name)
            ee_ct_forces = self._frankas[env_idx].get_ee_ct_forces(env_idx, self._franka_name)

            all_obs[env_idx, :7] = all_joints[:7]
            all_obs[env_idx, 7] = all_joints[-1] * 2 # gripper width is 2 * each gripper's prismatic length
            all_obs[env_idx, 8:15] = transform_to_np(ee_transform, format='wxyz')
            l_finger_pose_transform, r_finger_pose_transform = self._frankas[env_idx].get_finger_transforms(env_idx, self._franka_name)
            self.l_finger_pos[env_idx] = transform_to_np(l_finger_pose_transform, format='wxyz')[:3]
            self.r_finger_pos[env_idx] = transform_to_np(r_finger_pose_transform, format='wxyz')[:3]
            all_obs[env_idx, 15:18] = ee_ct_forces
            # all_obs[env_idx, 18:21] = ee_ct_forces

        return all_obs

    def _compute_rews(self, all_obs, all_actions):
        return np.zeros(self.n_envs)

    def _compute_dones(self, all_obs, all_actions, all_rews):
        return np.zeros(self.n_envs)

    def _reset(self, env_idxs):
        if not self._has_first_reset:
            self._init_joints = []
            for env_idx in env_idxs:
                self._init_joints.append(self._frankas[env_idx].get_joints(env_idx, self._franka_name))

        for env_idx in env_idxs:
            self._frankas[env_idx].set_joints(env_idx, self._franka_name, self._init_joints[env_idx])
            self._frankas[env_idx].set_joints_targets(env_idx, self._franka_name, self._init_joints[env_idx])

            if self._action_mode == 'joints':
                if 'randomize_joints' in self._cfg['franka'] and self._cfg['franka']['randomize_joints']:
                    init_random_joints = np.clip(np.random.normal(self._init_joints[env_idx], \
                        (self._frankas[env_idx].joint_limits_upper - self._frankas[env_idx].joint_limits_lower)/10), self._frankas[env_idx].joint_limits_lower, \
                        self._frankas[env_idx].joint_limits_upper)
                    self._frankas[env_idx].set_joints(env_idx, self._franka_name, init_random_joints)            
                    self._frankas[env_idx].set_joints_targets(env_idx, self._franka_name, init_random_joints)

            if self._action_mode == 'vic':
                self._frankas[env_idx].set_ee_transform(
                    env_idx, self._franka_name, self._init_ee_transforms[env_idx]
                )

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM

class GymFrankaBlockVecEnv(GymFrankaVecEnv):

    def _setup_single_env_gen(self, cfg):
        parent_setup = super()._setup_single_env_gen(cfg)
        self._block = GymBoxAsset(self._scene, **cfg['block']['dims'],
                            shape_props=cfg['block']['shape_props'],
                            rb_props=cfg['block']['rb_props'],
                            asset_options=cfg['block']['asset_options']
                            )
        self._banana = GymURDFAsset(
                            cfg['banana']['urdf_path'],
                            self._scene,
                            shape_props=cfg['banana']['shape_props'],
                            rb_props=cfg['banana']['rb_props'],
                            asset_options=cfg['banana']['asset_options']
                            )
        self._block_name = 'block'
        self._banana_name = 'banana0'

        def setup(scene, env_idx):
            parent_setup(scene, env_idx)
            scene.add_asset(self._block_name, self._block, gymapi.Transform())
            scene.add_asset(self._banana_name, self._banana, gymapi.Transform())
        return setup
    
    def parse_sim_params(self, physics_engine, config_sim) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params
        
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = self.gym.create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _reset(self, env_idxs):
        super()._reset(env_idxs)
        for env_idx in env_idxs:
            block_pose = gymapi.Transform(
                p=gymapi.Vec3(
                    (np.random.rand() * 2 - 1) * 0.1 + 0.5,
                    np.random.rand() * 0.2,
                    self._cfg['table']['dims']['sz'] + self._cfg['block']['dims']['sz'] / 2 + 0.05
                ))

            banana_pose = gymapi.Transform(
                p=gymapi.Vec3(
                    (np.random.rand() * 2 - 1) * 0.1 + 0.5,
                    -np.random.rand() * 0.2,
                    self._cfg['table']['dims']['sz'] + 0.05
                ),
                r=rpy_to_quat([np.pi/2, np.pi/2, -np.pi/2])
                )

            self._block.set_rb_transforms(env_idx, self._block_name, [block_pose])
            self._banana.set_rb_transforms(env_idx, self._banana_name, [banana_pose])
            self.progress_buf[env_idxs] = 0

    def _init_obs_space(self, cfg):
        obs_space = super()._init_obs_space(cfg)

        # add pose of block to obs_space
        limits_low = np.concatenate([
            obs_space.low,
            [-10] * 3 + [-1] * 4
        ])
        limits_high = np.concatenate([
            obs_space.high,
            [10] * 3 + [1] * 4
        ])
        new_obs_space = Box(limits_low, limits_high, dtype=np.float32)

        return new_obs_space

    def _compute_obs(self, all_actions):
        all_obs = super()._compute_obs(all_actions)

        box_pose_obs = np.zeros((self.n_envs, 7))

        for env_idx in self._scene.env_idxs:
            block_transform = self._block.get_rb_transforms(env_idx, self._block_name)[0]
            box_pose_obs[env_idx, :] = transform_to_np(block_transform, format='wxyz')

        all_obs = np.c_[all_obs, box_pose_obs]
        return all_obs
        
    def _compute_rews(self, all_obs, all_actions):
        
        # Compute per-env physical parameters

        # distance from hand to the cubeA
        d = np.linalg.norm(all_obs[:,18:21] - all_obs[:,8:11], -1)
        d_lf = np.linalg.norm(all_obs[:,18:21] - self.l_finger_pos, -1)
        d_rf = np.linalg.norm(all_obs[:,18:21] - self.r_finger_pos, -1)
        dist_reward = 1 - np.tanh(10.0 * (d + d_lf + d_rf) / 3)

        print(d)

        # reward for lifting cubeA
        cubeA_height = all_obs[:,20] - 0.5 #0.5 - table height
        cubeA_lifted = (cubeA_height - 0.05) > 0.04
        lift_reward = cubeA_lifted

        # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
        # offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
        # offset[:, 2] = (cubeA_size + cubeB_size) / 2
        # d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
        # align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted
        #
        # # Dist reward is maximum of dist and align reward
        # dist_reward = torch.max(dist_reward, align_reward)
        #
        # # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
        # cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
        # cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
        # gripper_away_from_cubeA = (d > 0.04)
        # stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA
        #
        # # Compose rewards
        #
        # # We either provide the stack reward or the align + dist reward
        # rewards = torch.where(
        #     stack_reward,
        #     reward_settings["r_stack_scale"] * stack_reward,
        #     reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
        #         "r_align_scale"] * align_reward,
        # )
        self.progress_buf += 1

        return 0.1 * dist_reward + 1.5 * lift_reward

    def _compute_dones(self, all_obs, all_actions, all_rews):
        reset_buf = np.zeros(self.n_envs)
        reset_buf = np.where((self.progress_buf >= 500 - 1),
                                np.ones_like(reset_buf), reset_buf)
        return reset_buf
