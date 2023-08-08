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

class Ant(Base, VecEnv):

    def __init__(self, runs_dir, tasks, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 8


        self.tasks = torch.tensor(tasks)
        self.num_tasks = len(self.tasks)

        Base.__init__(self, runs_dir=runs_dir, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        if self.viewer != None:
            cam_pos = gymapi.Vec3(5.0, 5.0, 2.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

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
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
        asset_file = "mjcf/nv_ant.xml"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        extremity_indices = [self.gym.find_asset_rigid_body_index(ant_asset, name) for name in extremity_names]
        sensor_pose = gymapi.Transform()
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)

        self.ant_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0], extremity_names[i])

    def _update_states(self):
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self.states.update({
            # Franka
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
        self.rew_buf[:] = self.compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = self.compute_ant_observations(
        self.obs_buf, self.root_states, self.targets, self.potentials,
        self.inv_start_rot, self.dof_pos, self.dof_vel,
        self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
        self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale,
        self.basis_vec0, self.basis_vec1, self.up_axis_idx)

    def reset_process(self, env_ids):

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


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
        self.actions = actions.clone().to(self.device)
        forces = self.actions * self.joint_gears * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            # for env_id in env_ids:
            # self.episode_count += 1
            # print()
            # self.writer.add_scalar("Total Reward per episode", torch.sum(self.reward_per_episode)/self.num_envs, self.episode_count)
            # # self.writer.add_scalar("Episode Length/Task {}".format(self.tasks[env_id]), self.progress_buf[env_id], self.episode_count[env_id])
            # self.reward_per_episode = 0
            self.reset_process(env_ids)

        self.compute_observations()
        self.compute_reward()
        # self.reward_per_episode += self.compute_franka_reward()

    def compute_ant_reward(self,
            obs_buf,
            reset_buf,
            progress_buf,
            actions,
            up_weight,
            heading_weight,
            potentials,
            prev_potentials,
            actions_cost_scale,
            energy_cost_scale,
            joints_at_limit_cost_scale,
            termination_height,
            death_cost,
            max_episode_length
    ):

        # reward from direction headed
        heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
        heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

        # aligning up axis of ant and environment
        up_reward = torch.zeros_like(heading_reward)
        up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

        # energy penalty for movement
        actions_cost = torch.sum(actions ** 2, dim=-1)
        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
        dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

        # reward for duration of staying alive
        alive_reward = torch.ones_like(potentials) * 0.5
        progress_reward = potentials - prev_potentials

        total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
                       actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale

        # adjust reward for fallen agents
        total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost,
                                   total_reward)

        # reset agents
        reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

        self.reset_buf = reset

        return total_reward

    def compute_ant_observations(self, obs_buf, root_states, targets, potentials,
                                 inv_start_rot, dof_pos, dof_vel,
                                 dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                 sensor_force_torques, actions, dt, contact_force_scale,
                                 basis_vec0, basis_vec1, up_axis_idx):

        torso_position = root_states[:, 0:3]
        torso_rotation = root_states[:, 3:7]
        velocity = root_states[:, 7:10]
        ang_velocity = root_states[:, 10:13]

        to_target = targets - torso_position
        to_target[:, 2] = 0.0

        prev_potentials_new = potentials.clone()
        potentials = -torch.norm(to_target, p=2, dim=-1) / dt

        torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
            torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

        vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
            torso_quat, velocity, ang_velocity, targets, torso_position)

        dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

        # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
        obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                         yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                         up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                         dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 24) * contact_force_scale,
                         actions), dim=-1)

        return obs, potentials, prev_potentials_new, up_vec, heading_vec