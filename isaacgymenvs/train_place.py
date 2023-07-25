from object_manipulation_cube_GCRL_place import ObjManipulationCube
import random
import numpy as np

from stable_baselines3 import PPO, DDPG, SAC
from sb3_contrib import TQC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
import time

cfg = {'name': 'FrankaCubeStack', 'physics_engine': 'physx', 'env': {'numEnvs': 256, 'envSpacing': 1.5, 'episodeLength': 300, 'enableDebugVis': False, 'clipObservations': 5.0, 'clipActions': 1.0, 'startPositionNoise': 0.25, 'startRotationNoise': 0.785, 'frankaPositionNoise': 0.0, 'frankaRotationNoise': 0.0, 'frankaDofNoise': 0.25, 'aggregateMode': 3, 'actionScale': 1.0, 'distRewardScale': 0.1, 'liftRewardScale': 1.5, 'alignRewardScale': 16.0, 'stackRewardScale': 16.0, 'controlType': 'osc', 'asset': {'assetRoot': '../../assets', 'assetFileNameFranka': 'urdf/franka_description/robots/franka_panda_gripper.urdf'}, 'enableCameraSensors': False}, 'sim': {'dt': 0.01667, 'substeps': 2, 'up_axis': 'z', 'use_gpu_pipeline': True, 'gravity': [0.0, 0.0, -9.81], 'physx': {'num_threads': 4, 'solver_type': 1, 'use_gpu': True, 'num_position_iterations': 8, 'num_velocity_iterations': 1, 'contact_offset': 0.005, 'rest_offset': 0.0, 'bounce_threshold_velocity': 0.2, 'max_depenetration_velocity': 1000.0, 'default_buffer_size_multiplier': 5.0, 'max_gpu_contact_pairs': 1048576, 'num_subscenes': 4, 'contact_collection': 0}}, 'task': {'randomize': False}}

success_grid = []#np.zeros((6,6,5))

env = ObjManipulationCube(cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0, headless=False, virtual_screen_capture=False, force_render=True)


eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000)

model = TQC(
    "MultiInputPolicy",
    env,
    batch_size=2048,
    buffer_size=1_000_000,
    gamma=0.95,
    learning_rate=0.001,
    learning_starts=1000,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
    tau=0.05,
    seed=3157870761,
    verbose=1
)

model = TQC.load("./logs/best_model.zip", env=env)

max_attempts = 1

for i in range(0,6):
    for j in range(0, 6):
        for k in range(0, 5):

            print("Current Test Region:", i / 10 - 0.3, j / 10 - 0.3, k / 10)
            print("Indices:", i, j, k)

            for _ in range(0, 10):
                x = random.uniform(i/10 - 0.3, i/10 - 0.3 + 0.1)
                y = random.uniform(j/10 - 0.3, j/10 - 0.3 + 0.1)
                z = random.uniform(k/10, k/10 + 0.1)

                # env = ObjManipulationCube(x, y, z, cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0, headless=False, virtual_screen_capture=False, force_render=True)



                # # Save a checkpoint every 1000 steps
                # checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./logs/',
                #                                          name_prefix='franka_goal')

                # print(env.reset())
                # quit()


                # model = PPO("MultiInputPolicy", env, verbose=1)

                test=True

                if test==False:
                    # model = TQC.load("./logs/best_model.zip", env=env)
                    model.learn(total_timesteps=100000000000000000000000000, callback=eval_callback)

                else:

                    obs, info = env.reset(x,y,z)
                    rews = 0
                    success = 0
                    num_attempts = 0

                    for _ in range(10000000000000000000):
                        action, _ = model.predict(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        rews += reward
                        if terminated or truncated:
                            obs, info = env.reset(x,y,z)
                            if terminated:
                                # print("Success - Reward : ", rews)
                                success=1
                                break
                            else:
                                # print("Time Over")
                                break
                            rews = 0
                            t = 0
                    success_grid.append([x,y,z,success])

success_grid = np.array(success_grid)
np.save("./logs/success_grid", success_grid)