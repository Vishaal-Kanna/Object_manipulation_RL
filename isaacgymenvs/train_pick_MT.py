from object_manipulation_cube_pick_MT import ObjManipulationCube

cfg = {'name': 'FrankaCubeStack', 'physics_engine': 'physx', 'env': {'numEnvs': 5, 'envSpacing': 1.5, 'episodeLength': 300, 'enableDebugVis': False, 'clipObservations': 5.0, 'clipActions': 1.0, 'startPositionNoise': 0.25, 'startRotationNoise': 0.785, 'frankaPositionNoise': 0.0, 'frankaRotationNoise': 0.0, 'frankaDofNoise': 0.25, 'aggregateMode': 3, 'actionScale': 1.0, 'distRewardScale': 0.1, 'liftRewardScale': 1.5, 'alignRewardScale': 16.0, 'stackRewardScale': 16.0, 'controlType': 'osc', 'asset': {'assetRoot': '../../assets', 'assetFileNameFranka': 'urdf/franka_description/robots/franka_panda_gripper.urdf'}, 'enableCameraSensors': False}, 'sim': {'dt': 0.01667, 'substeps': 2, 'up_axis': 'z', 'use_gpu_pipeline': True, 'gravity': [0.0, 0.0, -9.81], 'physx': {'num_threads': 4, 'solver_type': 1, 'use_gpu': True, 'num_position_iterations': 8, 'num_velocity_iterations': 1, 'contact_offset': 0.005, 'rest_offset': 0.0, 'bounce_threshold_velocity': 0.2, 'max_depenetration_velocity': 1000.0, 'default_buffer_size_multiplier': 5.0, 'max_gpu_contact_pairs': 1048576, 'num_subscenes': 4, 'contact_collection': 0}}, 'task': {'randomize': False}}

env = ObjManipulationCube('runs/Cube_MT', [1, 2, 3, 4, 5], cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0, headless=False, virtual_screen_capture=False, force_render=True)

from stable_baselines3 import PPO, DDPG, SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(env, best_model_save_path='./logs/Cube_MT',
                             log_path='./logs/Cube_MT', eval_freq=10000)

# # Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./logs/',
#                                          name_prefix='franka_goal')

# print(env.reset())
# quit()
model = TQC(
    "MlpPolicy",
    env,
    # batch_size=2048,
    buffer_size=1_000_000,
    gamma=0.95,
    learning_rate=0.0001,
    # learning_starts=1000,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    tau=0.05,
    seed=3157870761,
    verbose=1,
)

# model = PPO("MultiInputPolicy", env, verbose=1)

test=False

if test==False:
    # model = TQC.load("./logs/best_model.zip", env=env)
    model.learn(total_timesteps=100000000000000000000000000, callback=eval_callback)

else:
    model = TQC.load("/home/vishaal/omniverse/new_1/Object_manipulation_RL/isaacgymenvs/logs/Cube_MT/best_model.zip", env=env)

    obs = env.reset()
    rews = 0
    for _ in range(100000000000):
        action, _ = model.predict(obs)
        obs, reward, terminated, _ = env.step(action)
        # rews += reward
        # if terminated or truncated:
        #     obs = env.reset()
        #     if terminated:
        #         print("Success - Reward : ", rews)
        #     else:
        #         print("Time Over")
        #     rews = 0
        #     t = 0


env.close()