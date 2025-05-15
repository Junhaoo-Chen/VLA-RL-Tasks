import sapien
import torch
import numpy as np
from typing import Dict, Any, Union, List
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig

#####
##### !!!!!
##### INCOMPLETE 
##### !!!!!
#####


@register_env("KeyRotate-v1", max_episode_steps=100)
class KeyRotateEnv(BaseEnv):
    """KeyRotate
        Task: insert a key into another object and rotate it
        Goal: focus on the
        
        Robot: Panda
        Object: Retangle-liked object, a hollow object
        Scene: Tabletop
        Action: joint position / trajectory
    """
    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        # Ensure the scene is passed correctly to TableSceneBuilder
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            frame_init_p = torch.zeros((b, 3))

        

    def evaluate(self):
        """Determine success/failure of the task"""
        with torch.device(self.device):
            pass

        

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            # goal_pos=self.,
        )
        
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        with torch.device(self.device):
            # Initialize reward as a 1D tensor with shape (self.num_envs,)
            reward = torch.zeros(self.num_envs, device=self.device)


            return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Normalize the dense reward"""
        max_reward = (
            13.0  # Maximum possible reward (success + all intermediate rewards)
        )
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )

        # Side view camera
        side_camera = CameraConfig(
            "side_camera",
            pose=look_at(eye=[0.5, 0, 0.5], target=[0, 0, 0.2]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
        return [top_camera, side_camera]

    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human viewing"""
        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[0.6, 0.6, 0.6], target=[0, 0, 0.1]),
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )