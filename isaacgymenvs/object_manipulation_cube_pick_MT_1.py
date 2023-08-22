# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import random
import time

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from task_base_class_pick_MT import Base
import torch

from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

import math

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0
    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class ObjManipulationCube(Base, VecEnv):

    def __init__(self, runs_dir, tasks, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.test = False

        self.tasks = torch.tensor(tasks)

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 16+1 if self.control_type == "osc" else 23+1
        self.cfg["env"]["numGoals"] = 12
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 4 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.tasks = torch.tensor(tasks)
        self.num_tasks = len(self.tasks)

        Base.__init__(self, runs_dir=runs_dir, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.const_actions = torch.zeros((self.num_envs), device=self.device)
        self.time_ = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([50.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([1.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        self.reward_per_episode = torch.zeros(self.num_envs, device=self.device)
        self.lift_reward_per_episode = torch.zeros(self.num_envs, device=self.device)
        self.dist_reward_per_episode = torch.zeros(self.num_envs, device=self.device)
        self.episode_count = torch.zeros(self.num_envs, device=self.device)

        self.writer = SummaryWriter(log_dir=runs_dir)

        # Refresh tensors
        self._refresh()

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def seed(self, seed):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            n = self.num_envs
        else:
            n = len(indices)
        return [False] * n

    def env_method(self, method_name, *method_args, indices = None, **method_kwargs):
        pass
        """Call instance methods of vectorized environments."""
        # target_envs = self._get_target_envs(indices)
        # return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        # indices = self._get_indices(indices)
        # return [self.envs[i] for i in indices]
        pass

    def close(self):
        pass

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 0.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, table_pos[2] + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # self.cubeA_size = 0.050
        cubeA_asset = []

        items = ["025_mug", "010_potted_meat_can", "cube", "cube_shell", "factory_bolt_m20_tight"]

        for i, asset in enumerate(items):
            asset_root = "../assets"
            box_asset_file = "MT_urdfs/" + asset + ".urdf"
            print("Task ", i+1, " : ", asset)

            # nut_asset = self.gym.load_asset(self.sim, urdf_root, nut_file, nut_options)

            cubeA_opts = gymapi.AssetOptions()
            cubeA_asset.append(self.gym.load_asset(self.sim, asset_root, box_asset_file, cubeA_opts))
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, table_pos[2] + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(0.0, 0.0, self._table_surface_pos[2])
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 3     # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 3     # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset[i], cubeA_start_pose, "cubeA", i, 2, 0)
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._cubeA_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        # self._cubeA_state = self._root_state[:, self.self.table_actor, :]
        # print(self._cubeA_state)
        # print(self._cubeA_state)
        self.base_pose = torch.zeros_like(self._root_state[:, 3, :])
        self.base_pose[:, 0] = self._root_state[:, 3, 0]
        self.base_pose[:, 1] = self._root_state[:, 3, 1]

        self.task_ids = self.tasks.tile(math.ceil(self.num_envs / self.num_tasks,)).view(self.num_envs, 1).cuda()
        # self.task_ids = torch.tensor([[5],
        #         [5],
        #         [5],
        #         [5],
        #         [5]], device='cuda:0')

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)
    def _update_states(self):
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self.states.update({
            # Franka
            "task_id": self.task_ids,
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
        })

    def _refresh(self):
        # print(self._eef_state[0,:3],)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self):

        self.rew_buf, _, _ = self.compute_franka_reward()

    def compute_observations(self):
        self._refresh()
        obs = ["task_id", "cubeA_quat", "cubeA_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        # print(self.states['cubeA_pos'])

        return self.obs_buf

    def reset_process(self, env_ids):

        env_ids_int32 = env_ids.to(dtype=torch.long)

        # print('---------------------------------------------------')
        if self.test==True:
            self._cubeA_state = torch.zeros((self.num_envs, 13), device=self.device)
            self._cubeA_state[:, 0] = 0.1#random.uniform(-0.2, 0.2)
            self._cubeA_state[:, 1] = 0.0#random.uniform(-0.2, 0.2)
            self._cubeA_state[:, 2] = self._table_surface_pos[2] + 0.05 / 2
            self._cubeA_state[:, 3] = 0
            self._cubeA_state[:, 4] = 0
            self._cubeA_state[:, 5] = 0
            self._cubeA_state[:, 6] = 1

            self._cubeA_state[:, 7] = 0
            self._cubeA_state[:, 8] = 0
            self._cubeA_state[:, 9] = 0
            self._cubeA_state[:, 10] = 0
            self._cubeA_state[:, 11] = 0
            self._cubeA_state[:, 12] = 0

            self.goal_position = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_position[:, 0] = 0.0
            self.goal_position[:, 1] = -0.3
            self.goal_position[:, 2] = self._table_surface_pos[2] + 0.05 / 2 + 0.2

        else:
            self._cubeA_state = torch.zeros((self.num_envs, 13), device=self.device)
            self._cubeA_state[:, 0] = torch.distributions.uniform.Uniform(-0.3, 0.3).sample([self.num_envs]).cuda() #torch.FloatTensor(self.num_envs).uniform_(-0.3, 0.3).cuda() #random.uniform(-0.3, 0.3)
            self._cubeA_state[:, 1] = torch.distributions.uniform.Uniform(-0.3, 0.3).sample([self.num_envs]).cuda()
            self._cubeA_state[:, 2] = self._table_surface_pos[2] + 0.05/2
            self._cubeA_state[:, 3] = 0
            self._cubeA_state[:, 4] = 0
            self._cubeA_state[:, 5] = 0
            self._cubeA_state[:, 6] = 1

            self._cubeA_state[:, 7] = 0
            self._cubeA_state[:, 8] = 0
            self._cubeA_state[:, 9] = 0
            self._cubeA_state[:, 10] = 0
            self._cubeA_state[:, 11] = 0
            self._cubeA_state[:, 12] = 0

            self.goal_position = torch.zeros((self.num_envs, 3), device=self.device)
            self.goal_position[:, 0] = random.uniform(-0.3, 0.3)
            self.goal_position[:, 1] = random.uniform(-0.3, 0.3)
            self.goal_position[:, 2] = self._table_surface_pos[2] + 0.05 / 2 + random.uniform(0.0, 0.5)

            # while torch.norm(self._cubeA_state[0, 0:3] - torch.tensor(torch.tensor(self.goal_position)).cuda(), dim=-1) < 0.1:
            #     self.goal_position = [random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),
            #                           self._table_surface_pos[2] + 0.05 / 2 + random.uniform(0.0, 0.5)]

        # Reset agent
        reset_noise = torch.zeros((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        self._q[env_ids, :] = pos[env_ids, :]
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos[env_ids, :]
        self._effort_control[env_ids, :] = torch.zeros_like(pos)[env_ids, :]

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        self._root_state[env_ids, 3, :] = self.base_pose[env_ids] + self._cubeA_state[env_ids]

        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1].flatten()

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone()

        self.const_actions = torch.where(self.time_ == 0, self.actions[:, -1], self.const_actions)

        self.actions[:, -1] = self.const_actions
        self.time_ += 1
        self.time_ = torch.where(self.time_ > 20, 0, self.time_)
        # print(self.actions)

        # Split arm and gripper command

        u_arm = torch.zeros((self.num_envs, 6), device=self.device)
        u_arm[:, :3], u_gripper = self.actions[:, :-1], self.actions[:, -1]*0.05

        # Control arm (scale value first)
        # u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm



        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        fingers_width = self.states["q"][:, -2] + self.states["q"][:, -1]
        target_fingers_width = fingers_width + u_gripper.cuda()
        u_fingers[:, 0] = target_fingers_width / 2.0
        u_fingers[:, 1] = target_fingers_width / 2.0
        # u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
        #                               self.franka_dof_lower_limits[-2].item())
        # u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
        #                               self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            for env_id in env_ids:
                self.episode_count += 1
                self.writer.add_scalar("Total Reward per episode/Task {}".format(self.tasks[env_id]), self.reward_per_episode[env_id], self.episode_count[env_id])
                self.writer.add_scalar("Lift Reward per episode/Task {}".format(self.tasks[env_id]), self.lift_reward_per_episode[env_id], self.episode_count[env_id])
                self.writer.add_scalar("Distance Reward per episode/Task {}".format(self.tasks[env_id]), self.dist_reward_per_episode[env_id], self.episode_count[env_id])
                self.writer.add_scalar("Episode Length/Task {}".format(self.tasks[env_id]), self.progress_buf[env_id], self.episode_count[env_id])
                self.reward_per_episode[env_id] = 0
                self.lift_reward_per_episode[env_id] = 0
                self.dist_reward_per_episode[env_id] = 0
            self.reset_process(env_ids)

        self.compute_observations()
        self.compute_reward()
        rew_per_ep, dist_rew_per_ep, lift_rew_per_ep = self.compute_franka_reward()
        self.reward_per_episode += rew_per_ep
        self.lift_reward_per_episode += lift_rew_per_ep
        self.dist_reward_per_episode += dist_rew_per_ep

    def compute_franka_reward(self):

        dist_reward = torch.norm(self.states['cubeA_pos'].cuda() - self.states['eef_pos'], dim=-1)

        cubeA_height = self.states['cubeA_pos'][:, 2] - self.reward_settings["table_height"]
        cubeA_lifted = (cubeA_height - 0.05) > 0.04
        lift_reward = cubeA_lifted.cuda()



        reward = torch.where(dist_reward > 0.1, -dist_reward, lift_reward)

        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))


        # reset = torch.where(torch.abs(self.states['eef_pos'][:, 0]) >= 0.5, torch.ones_like(self.reset_buf), reset)
        # reset = torch.where(torch.abs(self.states['eef_pos'][:, 1]) >= 0.5, torch.ones_like(self.reset_buf), reset)
        #
        #
        # reset_buf = torch.where(abs(self.states['cubeA_pos'][:, 0]) >= 0.5, torch.ones_like(reset_buf), reset_buf)
        # reset_buf = torch.where(abs(self.states['cubeA_pos'][:, 1]) >= 0.5, torch.ones_like(reset_buf), reset_buf)

        self.reset_buf = reset

        return reward, -dist_reward, lift_reward

# @torch.jit.script
# def compute_franka_reward(achiev):
#     # Compute per-env physical parameters
#     # target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
#     # cubeA_size = states["cubeA_size"]
#     # cubeB_size = states["cubeB_size"]
#
#     # distance from hand to the cubeA
#     # d = torch.norm(states["cubeA_pos_relative"], dim=-1)
#     # d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
#     # d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
#     # dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
#     #
#     # # reward for lifting cubeA
#     # cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
#     # cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
#     # lift_reward = cubeA_lifted
#     #
#     # # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
#     # offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
#     # offset[:, 2] = (cubeA_size) / 2
#     # d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
#     # align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted
#     #
#     # # Dist reward is maximum of dist and align reward
#     # dist_reward = torch.max(dist_reward, align_reward)
#     #
#     # # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
#     # cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
#     # cubeA_on_cubeB = torch.abs(cubeA_height) < 0.02
#     # gripper_away_from_cubeA = (d > 0.04)
#     # stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA
#     #
#     # # Compose rewards
#     #
#     # # We either provide the stack reward or the align + dist reward
#     # rewards = torch.where(
#     #     stack_reward,
#     #     reward_settings["r_stack_scale"] * stack_reward,
#     #     reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
#     #         "r_align_scale"] * align_reward,
#     # )
#
#     # Compute resets
#     # reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)
#
#     return rewards