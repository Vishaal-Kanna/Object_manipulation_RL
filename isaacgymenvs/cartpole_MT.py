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
import math

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from task_base_class_cartpole_MT import Base
import torch

from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

class Cartpole(Base, VecEnv):

    def __init__(self, runs_dir, tasks, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # self._scene = GymScene(cfg['scene'])

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 5
        self.cfg["env"]["numActions"] = 1

        self.tasks = torch.tensor(tasks)
        self.num_tasks = len(self.tasks)

        Base.__init__(self, runs_dir=runs_dir, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.reward_per_episode = torch.zeros(self.num_envs, device=self.device)
        self.episode_count = torch.zeros(self.num_envs, device=self.device)

        self.writer = SummaryWriter(log_dir=runs_dir)

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
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        task_ids = self.tasks.tile(math.ceil(self.num_envs / self.num_tasks,))

        cartpole_asset = []

        for i in range(self.num_envs):
            asset_root = "../assets"
            asset_file = "urdf/cartpole_{}.urdf".format(task_ids[i])

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True

            cartpole_asset.append(self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))

            self.num_dof = self.gym.get_asset_dof_count(cartpole_asset[i])


        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset[i], pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def compute_observations(self, env_ids=None):

        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        task_ids = self.tasks.tile(math.ceil(self.num_envs / self.num_tasks,))

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 4] = task_ids[env_ids].float()

        # self.obs_buf[0, 4] = 1
        # self.obs_buf[1, 4] = 2

        return self.obs_buf

    def pre_physics_step(self, actions):

        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def reset_process(self, env_ids):

        # print('------here----------')

        # if self.start==True:
        #     self.start=False
        # else:
        #     print('here1')
        #     # self.gym.destroy_viewer(self.viewer)
        #     print('here2')
        #     self.gym.destroy_sim(self.sim)
        #     print('here3')
        #     self.up_axis = self.cfg["sim"]["up_axis"]
        #
        #     self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        #     print('here4')
        #     # self._create_ground_plane()
        #     self._create_envs(1, self.cfg["env"]['envSpacing'], int(np.sqrt(1)))
        #     print('here5')


        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            for env_id in env_ids:
                self.episode_count[env_id] += 1
                self.writer.add_scalar("Total Reward per episode/Task {}".format(self.tasks[env_id]), self.reward_per_episode[env_id], self.episode_count[env_id])
                self.writer.add_scalar("Episode Length/Task {}".format(self.tasks[env_id]), self.progress_buf[env_id], self.episode_count[env_id])
                self.reward_per_episode[env_id] = 0
            self.reset_process(env_ids)

        self.observations = self.compute_observations()
        self.compute_reward()
        self.reward_per_episode += self.compute_cartpole_reward()


    def compute_reward(self):

        self.pole_angle = self.obs_buf[:, 2]
        self.pole_vel = self.obs_buf[:, 3]
        self.cart_vel = self.obs_buf[:, 1]
        self.cart_pos = self.obs_buf[:, 0]

        self.rew_buf = self.compute_cartpole_reward()

    def compute_cartpole_reward(self):

        reward = 1.0 - self.pole_angle * self.pole_angle - 0.01 * torch.abs(self.cart_vel) - 0.005 * torch.abs(self.pole_vel)

        # adjust reward for reset agents
        reward = torch.where(torch.abs(self.cart_pos) > self.reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(self.pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        reset = torch.where(torch.abs(self.cart_pos) > self.reset_dist, torch.ones_like(self.reset_buf), self.reset_buf)
        reset = torch.where(torch.abs(self.pole_angle) > np.pi / 2, torch.ones_like(self.reset_buf), reset)
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)

        self.reset_buf = reset

        return reward

    def compute_reward_for_logging(self, env_id):

        reward = 1.0 - self.pole_angle[env_id] * self.pole_angle[env_id] - 0.01 * torch.abs(self.cart_vel[env_id]) - 0.005 * torch.abs(self.pole_vel[env_id])

        # adjust reward for reset agents
        reward = torch.where(torch.abs(self.cart_pos[env_id]) > self.reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(self.pole_angle[env_id]) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        return reward