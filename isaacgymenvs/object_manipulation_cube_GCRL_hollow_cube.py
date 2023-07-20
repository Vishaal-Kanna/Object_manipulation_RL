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
from vec_task import Base
import torch

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


class ObjManipulationCube(Base):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.test = False

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
        self.cfg["env"]["numObservations"] = 16 if self.control_type == "osc" else 23
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

        self.cubeA_size = 0.1

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

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

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(1, self.cfg["env"]['envSpacing'], int(np.sqrt(1)))

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

        asset_root = "../assets"
        box_asset_file = "urdf/cube_shell.urdf"
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file, cubeA_opts)

        # Create cubeA asset
        # cubeA_opts = gymapi.AssetOptions()
        # cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
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


        env_ptr = self.gym.create_env(self.sim, lower, upper, 1)

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
        franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", 0, 0, 0)
        self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

        if self.aggregate_mode == 2:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        # Create table
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", 0, 1, 0)
        table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", 0, 1, 0)

        if self.aggregate_mode == 1:
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        # Create cubes
        self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", 0, 2, 0)
        self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)

        if self.aggregate_mode > 0:
            self.gym.end_aggregate(env_ptr)

        # Store the created env pointers
        self.env = env_ptr
        self.franka = franka_actor

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.env
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
        self.num_dofs = self.gym.get_sim_dof_count(self.sim)

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(1, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(1, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(1, -1, 13)
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

        # Initialize actions
        self._pos_control = torch.zeros((self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:7]
        self._gripper_control = self._pos_control[7:9]

        self._global_indices = torch.arange(1 * 4, dtype=torch.int32,
                                            device=self.device).view(1, -1)

    def _update_states(self):
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]

        tx = gymapi.Transform(gymapi.Vec3(self._cubeA_state[0,0], self._cubeA_state[0,1], self._cubeA_state[0,2]), gymapi.Quat(self._cubeA_state[0,3], self._cubeA_state[0,4], self._cubeA_state[0,5], self._cubeA_state[0,5]))
        # print('-------------------')
        # print(self._cubeA_state)
        # print(tx.transform_point(gymapi.Vec3(0.0, self.cubeA_size / 2, - self.cubeA_size / 2)))
        # print('-------------------')
        # # cube_A_pose = gymapi.Transform.from_buffer(poses)
        # # print(_rigid_body_state_tensor)
        # print(tx.transform_point(gymapi.Vec3(0.0, 0.05, -0.05 / 2)).x)
        # quit()
        txx = tx.transform_point(gymapi.Vec3(0.0, self.cubeA_size / 2, -self.cubeA_size)).x
        txy = tx.transform_point(gymapi.Vec3(0.0, self.cubeA_size / 2, -self.cubeA_size)).y
        txz = tx.transform_point(gymapi.Vec3(0.0, self.cubeA_size / 2, -self.cubeA_size)).z


        self.states.update({
            # Franka
            "q": self._q[0,:],
            "q_gripper": self._q[0,-2:],
            "eef_pos": self._eef_state[0,:3],
            "eef_quat": self._eef_state[0,3:7],
            "eef_vel": self._eef_state[0,7:],
            "eef_lf_pos": self._eef_lf_state[0,:3],
            "eef_rf_pos": self._eef_rf_state[0,:3],
            # Cubes
            "cubeA_quat": self._cubeA_state[0,3:7],
            "cubeA_pos": torch.tensor([txx, txy, txz]).cuda(), #self._cubeA_state[0,:3],
            "cubeA_pos_relative": self._cubeA_state[0,:3] - self._eef_state[0,:3],
        })

        # print(self.states["cubeA_pos"])

    def _refresh(self):
        # print(self._eef_state[0,:3],)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, achieved_goal, desired_goal, info=None):

        if not torch.is_tensor(achieved_goal):
            achieved_goal = torch.from_numpy(achieved_goal)

        if not torch.is_tensor(desired_goal):
            desired_goal = torch.from_numpy(desired_goal)

        rewards = self.compute_franka_reward(achieved_goal, desired_goal)

        if torch.is_tensor(rewards):
            rewards = rewards.detach().cpu().numpy()

        return rewards

    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]

        self.observations["observation"] = torch.cat([self.states[ob] for ob in obs], dim=-1)
        obs = ["cubeA_pos", "eef_pos", "eef_lf_pos", "eef_rf_pos"]
        # obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.observations["achieved_goal"] = torch.cat([self.states[ob] for ob in obs], dim=-1)
        self.observations["desired_goal"][:3] = torch.tensor(self.goal_position)
        self.observations["desired_goal"][3:6] = torch.tensor(self.states["cubeA_pos"])
        self.observations["desired_goal"][6:9] = torch.tensor(self.states["cubeA_pos"])
        self.observations["desired_goal"][9:12] = torch.tensor(self.states["cubeA_pos"])

        # print(self.states['cubeA_pos'])

        return self.observations

    def reset_process(self):

        # print('---------------------------------------------------')
        if self.test==True:
            self._cubeA_state = torch.zeros((1, 13), device=self.device)
            self._cubeA_state[0, 0] = 0.1#random.uniform(-0.2, 0.2)
            self._cubeA_state[0, 1] = 0.0#random.uniform(-0.2, 0.2)
            self._cubeA_state[0, 2] = self._table_surface_pos[2] + self.cubeA_size / 2
            self._cubeA_state[0, 3] = 0
            self._cubeA_state[0, 4] = 0
            self._cubeA_state[0, 5] = 0
            self._cubeA_state[0, 6] = 1

            self._cubeA_state[0, 7] = 0
            self._cubeA_state[0, 8] = 0
            self._cubeA_state[0, 9] = 0
            self._cubeA_state[0, 10] = 0
            self._cubeA_state[0, 11] = 0
            self._cubeA_state[0, 12] = 0

            self.goal_position = [0.3, 0.3, self._table_surface_pos[2] + self.cubeA_size / 2 + 0.2]

        else:
            self._cubeA_state = torch.zeros((1, 13), device=self.device)
            self._cubeA_state[0, 0] = random.uniform(-0.3, 0.3)
            self._cubeA_state[0, 1] = random.uniform(-0.3, 0.3)
            self._cubeA_state[0, 2] = self._table_surface_pos[2] #+ self.cubeA_size/2
            self._cubeA_state[0, 3] = 0
            self._cubeA_state[0, 4] = 0
            self._cubeA_state[0, 5] = 0
            self._cubeA_state[0, 6] = 1

            self._cubeA_state[0, 7] = 0
            self._cubeA_state[0, 8] = 0
            self._cubeA_state[0, 9] = 0
            self._cubeA_state[0, 10] = 0
            self._cubeA_state[0, 11] = 0
            self._cubeA_state[0, 12] = 0

            # print("self._cubeA_state",self._cubeA_state)


            self.goal_position = [random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),
                                  self._table_surface_pos[2] + self.cubeA_size / 2 + random.uniform(0.0, 0.5)]

            while torch.norm(self._cubeA_state[0,0:3] - torch.tensor(torch.tensor(self.goal_position)).cuda(), dim=-1) < 0.1:
                self.goal_position = [random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),
                                      self._table_surface_pos[2] + self.cubeA_size / 2 + random.uniform(0.0, 0.5)]

        # Reset agent
        reset_noise = torch.zeros((1, 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[0, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[0, :] = pos
        self._qd[0, :] = torch.zeros_like(self._qd)

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[:] = pos
        self._effort_control[:] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[0, 0].flatten()
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
                                              gymtorch.unwrap_tensor(torch.tensor([0], dtype=torch.int32).cuda()),
                                              len(torch.tensor([0], dtype=torch.int32).cuda()))

        self._root_state[0,3,:] = self._cubeA_state
        multi_env_ids_cubes_int32 = self._global_indices[0, -1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        # time.sleep(10)

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

        # print(self.actions)

        # Split arm and gripper command
        u_arm = torch.zeros(6, device=self.device)
        # print(self.actions[:-1])
        u_arm[:3], u_gripper = self.actions[:-1], self.actions[-1]
        # u_arm[4] = self.actions[-2]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        # u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:] = u_arm

        # print(self._effort_control)

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        fingers_width = self.states["q"][-2]+self.states["q"][-1]
        target_fingers_width = fingers_width + u_gripper
        u_fingers[0] = target_fingers_width / 2.0
        u_fingers[1] = target_fingers_width / 2.0
        # u_fingers[0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
        #                               self.franka_dof_lower_limits[-2].item())
        # u_fingers[1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
        #                               self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.observations = self.compute_observations()
        self.rewards = self.compute_reward(self.observations["achieved_goal"], self.observations["desired_goal"]) #self._reward(self.reward_settings, self.states, self.observations["achieved_goal"], self.observations["desired_goal"])
        return self.observations, self.rewards

        # debug viz
        # if self.viewer and self.debug_viz:
        #     self.gym.clear_lines(self.viewer)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)
        #
        #     # Grab relevant states to visualize
        #     eef_pos = self.states["eef_pos"]
        #     eef_rot = self.states["eef_quat"]
        #     cubeA_pos = self.states["cubeA_pos"]
        #     cubeA_rot = self.states["cubeA_quat"]
        #     cubeB_pos = self.states["cubeB_pos"]
        #     cubeB_rot = self.states["cubeB_quat"]
        #
        #     # Plot visualizations
        #     for i in range(self.num_envs):
        #         for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
        #             px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        #             py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        #             pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
        #
        #             p0 = pos[i].cpu().numpy()
        #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
        #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
        #             self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


    def is_success(self, achieved_goal, desired_goal):
        # print(desired_goal.shape)
        # quit()

        d = torch.norm(achieved_goal[:3].cuda() - desired_goal[:3].cuda(), dim=-1)
        if d < 0.07:
            # time.sleep(5)
            b = True
        else:
            b = False
        return b

    def compute_franka_reward(self, achieved_goal, desired_goal):

        # reward = 1.0 - self.pole_angle * self.pole_angle - 0.01 * torch.abs(self.cart_vel) - 0.005 * torch.abs(self.pole_vel)
        #
        # if torch.abs(self.cart_pos) > self.reset_dist:
        #     reward = -2.0
        # if torch.abs(self.pole_angle) > np.pi / 2:
        #     reward = -2.0

        # d = torch.norm(achieved_goal - desired_goal, dim=-1)
        # reward = -d
        # print(achieved_goal.shape)

        # if achieved_goal.shape == torch.Size([80, 12]):
        #     d = torch.norm(achieved_goal[:, :3] - achieved_goal[:, 3:6], dim=-1)
        #     d_lf = torch.norm(achieved_goal[:, :3] - achieved_goal[:, 6:9], dim=-1)
        #     d_rf = torch.norm(achieved_goal[:, :3] - achieved_goal[:, 9:12], dim=-1)
        #     dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
        #
        #     # reward for lifting cubeA
        #     cubeA_height = achieved_goal[:, 2] - self.reward_settings["table_height"]
        #     cubeA_lifted = (cubeA_height - 0.05) > 0.04
        #     lift_reward = cubeA_lifted
        #     # print(desired_goal.shape)
        #     d_ab = torch.norm(achieved_goal[:, :3] - desired_goal[:, :3], dim=-1)
        #     align_reward = (1 - torch.tanh(10.0 * d_ab)) * lift_reward
        #
        #     dist_reward = torch.max(dist_reward, align_reward)
        #
        #     reward = self.reward_settings["r_dist_scale"] * dist_reward + self.reward_settings["r_align_scale"] * align_reward + self.reward_settings["r_lift_scale"] * lift_reward
        #
        # else:
        #     d = torch.norm(achieved_goal[:3] - achieved_goal[3:6], dim=-1)
        #     d_lf = torch.norm(achieved_goal[:3] - achieved_goal[6:9], dim=-1)
        #     d_rf = torch.norm(achieved_goal[:3] - achieved_goal[9:12], dim=-1)
        #     dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
        #
        #     # reward for lifting cubeA
        #     cubeA_height = achieved_goal[2] - self.reward_settings["table_height"]
        #     cubeA_lifted = (cubeA_height - 0.05) > 0.04
        #     lift_reward = cubeA_lifted
        #
        #     # print(achieved_goal)
        #     # print(desired_goal)
        #
        #     d_ab = torch.norm(achieved_goal[:3].cuda() - desired_goal[:3].cuda(), dim=-1)
        #     align_reward = (1 - torch.tanh(10.0 * d_ab))*lift_reward
        #
        #     dist_reward = torch.max(dist_reward, align_reward)
        #
        #     reward = self.reward_settings["r_dist_scale"] * dist_reward + self.reward_settings["r_align_scale"] * align_reward + self.reward_settings["r_lift_scale"] * lift_reward

        # d = torch.norm(achieved_goal.cuda() - desired_goal.cuda(), dim=-1)
        # reward = -d

        # print(achieved_goal.shape)
        if achieved_goal.shape == torch.Size([1638, 12]):
            d = torch.norm(achieved_goal[:, 3:6].cuda() - desired_goal[:, 3:6].cuda(), dim=-1)
            d_lf = torch.norm(desired_goal[:, 6:9].cuda() - achieved_goal[:, 6:9].cuda(), dim=-1)
            d_rf = torch.norm(desired_goal[:, 9:12].cuda() - achieved_goal[:, 9:12].cuda(), dim=-1)
            dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

            cubeA_height = achieved_goal[:, 2] - self.reward_settings["table_height"]
            cubeA_lifted = (cubeA_height - self.cubeA_size) > 0.04
            lift_reward = cubeA_lifted.cuda()

            d_ab = torch.norm(achieved_goal[:, :3].cuda() - desired_goal[:, :3].cuda(), dim=-1)
            align_reward = (1 - torch.tanh(10.0 * d_ab))*lift_reward

            dist_reward = torch.max(dist_reward, align_reward)

            reward = self.reward_settings["r_dist_scale"] * dist_reward + self.reward_settings[
                "r_align_scale"] * align_reward + self.reward_settings["r_lift_scale"] * lift_reward
        else:
            d = torch.norm(achieved_goal[3:6].cuda() - desired_goal[3:6].cuda(), dim=-1)
            d_lf = torch.norm(desired_goal[6:9].cuda() - achieved_goal[6:9].cuda(), dim=-1)
            d_rf = torch.norm(desired_goal[9:12].cuda() - achieved_goal[9:12].cuda(), dim=-1)
            dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

            cubeA_height = achieved_goal[2] - self.reward_settings["table_height"]
            cubeA_lifted = (cubeA_height - self.cubeA_size) > 0.04
            lift_reward = cubeA_lifted.cuda()

            d_ab = torch.norm(achieved_goal[:3].cuda() - desired_goal[:3].cuda(), dim=-1)
            align_reward = (1 - torch.tanh(10.0 * d_ab)) * lift_reward

            dist_reward = torch.max(dist_reward, align_reward)

            reward = self.reward_settings["r_dist_scale"] * dist_reward + self.reward_settings[
                "r_align_scale"] * align_reward + self.reward_settings["r_lift_scale"] * lift_reward

        return reward

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