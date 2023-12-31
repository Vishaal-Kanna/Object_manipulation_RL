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

import copy
from typing import Dict, Any, Tuple, List, Set
from collections import OrderedDict
from copy import deepcopy

import gym
import gymnasium
from gymnasium import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import torch
import numpy as np
import operator, random
from copy import deepcopy
from isaacgymenvs.utils.utils import nested_dict_get_attr, nested_dict_set_attr

from collections import deque

from stable_baselines3.common.vec_env import VecEnv

import sys

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM

class Base(gymnasium.Env, VecEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        self.num_observations = config["env"].get("numObservations", 0)
        self.num_actions = config["env"]["numActions"]

        self.action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        # self.observation_space = spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        self.observation_space = self.obs_space = spaces.Dict(
            {
                "observation": spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf),
                "achieved_goal": spaces.Box(np.ones(12) * -np.Inf, np.ones(12) * np.Inf),
                "desired_goal": spaces.Box(np.ones(12) * -np.Inf, np.ones(12) * np.Inf),
            }
        )

        self.observations = OrderedDict({"observation": torch.zeros(self.num_envs, self.num_observations),
                             "achieved_goal": torch.zeros(self.num_envs, 12),
                             "desired_goal": torch.zeros(self.num_envs, 12)})

        self.progress = self.max_episode_length*torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.truncated = torch.ones(self.num_envs, device=self.device, dtype=torch.long)


        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames = 0

        # super().__init__(config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs)
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        # for env_id in range(self.num_envs):
        #     self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(2.0, 5.0, 3.0)
                cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    def step(self, actions):

        actions = torch.from_numpy(actions).cuda()

        self.observation = self.compute_observations()
        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.observations, self.rewards = self.post_physics_step()

        self.observations["observation"] = torch.clamp(self.observations["observation"], -self.clip_obs, self.clip_obs)
        self.observations["achieved_goals"] = torch.clamp(self.observations["achieved_goal"], -self.clip_obs, self.clip_obs)
        self.observations["desired_goals"] = torch.clamp(self.observations["achieved_goal"], -self.clip_obs, self.clip_obs)

        obs_np = OrderedDict({"observation": self.observations["observation"].detach().cpu().numpy(),
                             "achieved_goal": self.observations["achieved_goal"].detach().cpu().numpy(),
                             "desired_goal": self.observations["desired_goal"].detach().cpu().numpy()})

        # if self.is_success(self.observations["achieved_goal"], self.observations["desired_goal"]):
        #     terminated = True
        # else:
        #     terminated = False

        self.terminated = self.is_success(self.observations["achieved_goal"], self.observations["desired_goal"])

        # info = {"is_success": self.terminated}

        self.progress += 1

        # if self.progress >= self.max_episode_length - 1:
        #     truncated = True
        # else:
        #     truncated = False

        # self.truncated = torch.where((self.progress >= self.max_episode_length - 1), torch.ones_like(self.truncated), torch.zeros_like(self.truncated))
        #
        # env_ids = torch.where(self.terminated>0 or self.truncated>0, torch.ones_like(self.truncated), torch.zeros_like(self.truncated))
        env_ids = (self.progress >= self.max_episode_length - 1) | (self.terminated != 0)

        # print(env_ids)

        info = self._compute_infos(obs_np, actions.detach().cpu().numpy(), self.rewards, env_ids.detach().cpu().numpy())


        if torch.any(env_ids):
            print("Current Reward : ", np.mean(self.rewards))
            # info[env_idx]['terminal_observation'] = obs
            self.reset()
        # for env_idx in range(self.num_envs):
        #     if env_ids[env_idx]==True:
        #         self.reset()

        # print(self.progress)
        # print(env_ids)



        return obs_np, self.rewards, env_ids.detach().cpu().numpy(), info #self.terminated.detach().cpu().numpy(), self.truncated.detach().cpu().numpy(), {} #info

    def _compute_infos(self, all_obs, all_actions, all_rews, all_dones):
        return [{} for _ in range(self.num_envs)]

    def reset(
        self,
        *,
        seed = None,
        options= None,
    ):

        # env_ids = torch.where(self.terminated>0 or self.truncated>0, torch.ones_like(self.truncated), torch.zeros_like(self.truncated))
        env_ids = (self.progress >= self.max_episode_length - 1) | (self.terminated != 0)
        # print('here---------------------------')

        self.reset_process(env_ids)

        self.progress[env_ids] = 0

        self.observations["observation"] = torch.clamp(self.observations["observation"], -self.clip_obs, self.clip_obs)
        self.observations["achieved_goal"] = torch.clamp(self.observations["achieved_goal"], -self.clip_obs, self.clip_obs)
        self.observations["desired_goal"] = torch.clamp(self.observations["desired_goal"], -self.clip_obs, self.clip_obs)

        obs_np = OrderedDict({"observation": self.observations["observation"].detach().cpu().numpy(),
                              "achieved_goal": self.observations["achieved_goal"].detach().cpu().numpy(),
                              "desired_goal": self.observations["desired_goal"].detach().cpu().numpy()})

        info = {"is_success": self.is_success(self.observations["achieved_goal"], self.observations["desired_goal"])}

        return obs_np

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
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

    def close(self):
        self.sim.close()

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def seed(self, seed):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            n = self.num_envs
        else:
            n = len(indices)
        return [False] * n
