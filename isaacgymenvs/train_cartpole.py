from cartpole_MT import Cartpole

cfg = {'name': 'Cartpole', 'physics_engine': 'physx', 'env': {'numEnvs': 1, 'envSpacing': 4.0, 'resetDist': 3.0, 'maxEffort': 400.0, 'clipObservations': 5.0, 'clipActions': 1.0, 'asset': {'assetRoot': '../../assets', 'assetFileName': 'urdf/cartpole.urdf'}, 'enableCameraSensors': False}, 'sim': {'dt': 0.0166, 'substeps': 2, 'up_axis': 'z', 'use_gpu_pipeline': False, 'gravity': [0.0, 0.0, -9.81], 'physx': {'num_threads': 4, 'solver_type': 1, 'use_gpu': True, 'num_position_iterations': 4, 'num_velocity_iterations': 0, 'contact_offset': 0.02, 'rest_offset': 0.001, 'bounce_threshold_velocity': 0.2, 'max_depenetration_velocity': 100.0, 'default_buffer_size_multiplier': 2.0, 'max_gpu_contact_pairs': 1048576, 'num_subscenes': 4, 'contact_collection': 0}}, 'task': {'randomize': False}}

env = Cartpole('runs/Cartpole_MT', [2], cfg, rl_device='cuda:0', sim_device='cuda:0', graphics_device_id=0, headless=False, virtual_screen_capture=False, force_render=True)

from stable_baselines3 import PPO, DDPG, SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(env, best_model_save_path='./logs/Cartpole_MT',
                             log_path='./logs/Cartpole_MT', eval_freq=1000)

model = TQC("MlpPolicy", env, verbose=1)

test=False

if test==False:
    # model = TQC.load("./logs/best_model.zip", env=env)
    model.learn(total_timesteps=70000)#, callback=eval_callback)
    model.save('./logs/Cartpole_MT/model_70000')

else:
    model = TQC.load("./logs/Cartpole_MT/model_70000.zip", env=env)

    obs, info = env.reset()
    rews = 0
    for _ in range(100000000000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        # rews += reward
        # if done:
        #     obs, info = env.reset()

env.close()