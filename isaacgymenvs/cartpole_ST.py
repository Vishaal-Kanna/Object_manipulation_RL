# Copyright (c) 2018-2023, NVIDIA Corporation
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
from task_base_class_cartpole import Base
import torch

class Cartpole(Base):

    def __init__(self, runs_dir, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # self._scene = GymScene(cfg['scene'])

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 1

        super().__init__(runs_dir=runs_dir, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(1, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(1, self.num_dof, 2)[..., 1]

        self.cart_goal_pose = random.uniform(-self.reset_dist + 1, self.reset_dist - 1)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(1, self.cfg["env"]['envSpacing'], int(np.sqrt(1)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = "../assets"
        asset_file = "urdf/cartpole.urdf"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        env_ptr = self.gym.create_env(self.sim, lower, upper, 1)
        cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", 0, 1, 0)

        dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
        dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0
        self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

        self.env = env_ptr
        self.cartpole_handle = cartpole_handle

        camera_props = gymapi.CameraProperties()
        camera_props.width = 256
        camera_props.height = 128
        self.camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

        self.gym.set_camera_location(self.camera_handle, env_ptr, gymapi.Vec3(3.0, 0.0, 2.0), gymapi.Vec3(0.0, 0.0, 2.5))

        self.env = env_ptr

    def compute_observations(self):
        # self.gym.render_all_camera_sensors(self.sim)
        # color_image = self.gym.get_camera_image(self.sim, self.env, self.camera_handle, gymapi.IMAGE_COLOR)

        self.gym.refresh_dof_state_tensor(self.sim)

        # self.observations["states"] = color_image.reshape(128,256,4)
        self.observations = torch.tensor([self.dof_pos[0, 0], self.dof_vel[0, 0], self.dof_pos[0, 1], self.dof_vel[0, 1]])

        self.pole_angle = self.observations[2]
        self.pole_vel = self.observations[3]
        self.cart_vel = self.observations[1]
        self.cart_pos = self.observations[0]

        return self.observations

    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(1 * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def reset_process(self):
        positions = 0.0 * (torch.rand((1, self.num_dof), device=self.device) - 0.5)
        velocities = 0.0 * 0.5 * (torch.rand((1, self.num_dof), device=self.device) - 0.5)

        self.dof_pos[0, :] = positions[:]
        self.dof_vel[0, :] = velocities[:]

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(torch.tensor([0], dtype=torch.int32).cuda()),
                                              len(torch.tensor([0], dtype=torch.int32).cuda()))
        self.cart_goal_pose = random.uniform(-self.reset_dist + 1, self.reset_dist - 1)

    def post_physics_step(self):

        self.observations = self.compute_observations()

        self.rewards = self.compute_reward()

        return self.observations, self.rewards, 0, 0

    def compute_reward(self):

        rewards = self.compute_cartpole_reward()

        if torch.is_tensor(rewards):
            rewards = rewards.detach().cpu().numpy()

        return rewards

    def compute_cartpole_reward(self):

        reward = 1.0 - self.pole_angle * self.pole_angle - 0.01 * torch.abs(self.cart_vel) - 0.005 * torch.abs(self.pole_vel)

        if torch.abs(self.cart_pos) > self.reset_dist:
            reward = -2.0
        if torch.abs(self.pole_angle) > np.pi / 2:
            reward = -2.0

        return reward