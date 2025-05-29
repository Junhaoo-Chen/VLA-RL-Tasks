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



@register_env("Test-v1", max_episode_steps=100)
class TestEnv(BaseEnv):
    """Test
    NOT A TASK !!!
    """
    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        # Ensure the scene is passed correctly to TableSceneBuilder
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        
        ### test code
        # Add the object to the scene
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[0.02] * 3,
        )
        builder.add_box_visual(
            half_size=[0.02] * 3,
            material=sapien.render.RenderMaterial(
                base_color=[1, 0, 0, 1],  # Red cube
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
        self.obj = builder.build(name="cube")
        # PushCube has some other code after this removed for brevity that 
        # spawns a goal object (a red/white target) stored at self.goal_region
        ### test code


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        ### test code
        with torch.device(self.device):
            # Set the initial pose of the object
            initial_pose = torch.tensor([[0.0, 0.0, 0.02]], device=self.device)
            self.obj.set_pose(sapien.Pose(p=initial_pose[0].detach().cpu().tolist()))  # Ensure proper tensor handling

            # Set the goal position for the task
            self.goal_position = torch.tensor([[0.2, 0.0, 0.02]], device=self.device)
        ### test code
        

    def evaluate(self):
        """Determine success/failure of the task"""
        with torch.device(self.device):
            # Calculate the distance between the object and the goal position
            object_position = torch.tensor(self.obj.pose.p, device=self.device).unsqueeze(0)  # Ensure batching
            distance_to_goal = torch.linalg.norm(object_position - self.goal_position, dim=-1)

            # Define a success threshold
            success_threshold = 0.05

            # Check if the object is within the success threshold
            success = (distance_to_goal < success_threshold).view(self.num_envs)  # Flatten to 1D
            # Return evaluation results
            return {"success": success, "distance_to_goal": distance_to_goal.view(self.num_envs)}
        

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_position,
        )
        
        return obs


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        with torch.device(self.device):
            # Initialize reward as a 1D tensor with shape (self.num_envs,)
            reward = torch.ones(self.num_envs, device=self.device)

            # Ensure success is a 1D tensor with shape (self.num_envs,)
            success = info["success"].view(self.num_envs)  # Flatten to match batch size

            # Update reward based on success
            reward = torch.where(success, reward + 10.0, reward)

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