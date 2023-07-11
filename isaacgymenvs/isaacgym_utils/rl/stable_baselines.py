'''
To make vec_env.GymVecEnv compatible with stable_baselines
'''

from stable_baselines3.common.vec_env import VecEnv
from .franka_vec_env import GymFrankaBlockVecEnv
from isaacgym import gymapi


class StableBaselinesVecEnvAdapter(VecEnv):

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


class GymFrankaBlockVecEnvStableBaselines(GymFrankaBlockVecEnv, StableBaselinesVecEnvAdapter):
    '''
    An example of how to convert a GymVecEnv to a StableBaselines-compatible VecEnv
    '''

    def __init__(self, *args, **kwargs):
        GymFrankaBlockVecEnv.__init__(self, *args, **kwargs)
        #self.cfg = {'name': 'FrankaCubeStack', 'physics_engine': 'physx', 'env': {'numEnvs': 64, 'envSpacing': 1.5, 'episodeLength': 300, 'enableDebugVis': False, 'clipObservations': 5.0, 'clipActions': 1.0, 'startPositionNoise': 0.25, 'startRotationNoise': 0.785, 'frankaPositionNoise': 0.0, 'frankaRotationNoise': 0.0, 'frankaDofNoise': 0.25, 'aggregateMode': 3, 'actionScale': 1.0, 'distRewardScale': 0.1, 'liftRewardScale': 1.5, 'alignRewardScale': 2.0, 'stackRewardScale': 16.0, 'controlType': 'joint_tor', 'asset': {'assetRoot': '../assets', 'assetFileNameFranka': 'urdf/franka_description/robots/franka_panda_gripper.urdf'}, 'enableCameraSensors': False}, 'sim': {'dt': 0.01667, 'substeps': 2, 'up_axis': 'z', 'use_gpu_pipeline': True, 'gravity': [0.0, 0.0, -9.81], 'physx': {'num_threads': 4, 'solver_type': 1, 'use_gpu': True, 'num_position_iterations': 8, 'num_velocity_iterations': 1, 'contact_offset': 0.005, 'rest_offset': 0.0, 'bounce_threshold_velocity': 0.2, 'max_depenetration_velocity': 1000.0, 'default_buffer_size_multiplier': 5.0, 'max_gpu_contact_pairs': 1048576, 'num_subscenes': 4, 'contact_collection': 0}}, 'task': {'randomize': False}}

        #self.gym = gymapi.acquire_gym()
        #self.device_id = 0
        #self.graphics_device_id = 0
        #self.physics_engine = "physx"
        #self.sim_params = super().parse_sim_params(self.physics_engine, self.cfg["sim"])
        #super().create_sim()
        #self.gym.prepare_sim(self.sim)
