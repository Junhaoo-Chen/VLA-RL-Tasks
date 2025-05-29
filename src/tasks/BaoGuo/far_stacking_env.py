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


BLOCK_SIZE = [0.05, 0.08, 0.01]  # half-sizes for x, y, z (m)

@register_env("FarStack-v1", max_episode_steps=100)
class FarStackEnv(BaseEnv):
    """FarStack
        Task: Stack the block on top of one another, try to push the boundary of the whole as far as possible
        Goal: understanding of the object interaction 
        
        Robot: Panda
        Object: cuboid blocks
        Scene: Tabletop
        Action: joint position / trajectory
    """
    def __init__(self, *args, robot_uids="panda", n_block=3, **kwargs):
        self.n_block = n_block
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with a table and blocks
        
        """
        # Ensure the scene is passed correctly to TableSceneBuilder
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        
        
        # Load the blocks with the specified material
        self.blocks = []
        for i in range(self.n_block):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=BLOCK_SIZE)
            color = [(i + 1) / self.n_block, 0.2, 0.8, 1.0]
            builder.add_box_visual(
                half_size=BLOCK_SIZE,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            builder.initial_pose = sapien.Pose(
                p=[0, 0, BLOCK_SIZE[2] + 0.5], q=[1, 0, 0, 0]
            )
            block = builder.build(name=f"block_{i}")
            self.blocks.append(block)

        
        # A marker position as the starting point for stacking
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=BLOCK_SIZE,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
        )
        builder.initial_pose = sapien.Pose(
            p=[0, 0, 0], q=[1, 0, 0, 0]
        )
        marker = builder.build_static(name="marker")
        self.marker = marker
        
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            start_pos = torch.zeros(b, 3, device=self.device)
            start_pos[..., 0] = torch.rand(b) * 0.4 - 0.2
            start_pos[..., 1] = torch.rand(b) * 0.4 - 0.2
            start_pos[..., 2] = 0.0
            start_pose = Pose.create_from_pq(p=start_pos, q=[1, 0, 0, 0])
            self.marker.set_pose(start_pose)
            
            for i, block in enumerate(self.blocks):
                init_pos = torch.zeros(b, 3, device=self.device)
                init_pos[..., 0] = torch.rand(b) * 0.4 - 0.2
                init_pos[..., 1] = torch.rand(b) * 0.4 - 0.2
                init_pos[..., 2] = BLOCK_SIZE[2] + 0.5
                init_pose = Pose.create_from_pq(p=init_pos, q=[1, 0, 0, 0])
                block.set_pose(init_pose)
                
                block.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                block.set_angular_velocity(torch.zeros((b, 3), device=self.device))
            
            
    def evaluate(self):
        """Determine success/failure of the task
            Description: 
                1. The stacking should be starting from the marker position
                2. Check if all blocks are stacked on top of each other within a certain threshold 
                    (physically, the distance between the centers of the blocks should not exceed a half of the block size)
                3. Calulate the horizontal distance from the center of the bottom block to the center of the top block         
        """
        with torch.device(self.device):
            # Get block poses
            block_poses = [block.pose.p for block in self.blocks]
            block_poses = torch.stack(block_poses, dim=0)  # (n_block, 3)
            # Marker pose
            marker_pos = self.marker.pose.p  # (3,)

            # 1. Check if the bottom block is close to the marker
            bottom_block_pos = block_poses[0]
            marker_dist = torch.norm(bottom_block_pos - marker_pos, dim=-1)
            marker_threshold = BLOCK_SIZE[0]  # within half block size

            # 2. Check stacking: each block center is above previous, and horizontally close
            stacked = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for i in range(1, self.n_block):
                prev = block_poses[i - 1]
                curr = block_poses[i]
                # Check vertical order
                vertical = curr[..., 2] > prev[..., 2] + BLOCK_SIZE[2] * 1.5
                # Check horizontal distance
                horizontal = torch.norm(curr[..., :2] - prev[..., :2], dim=-1) < BLOCK_SIZE[0]
                stacked &= vertical & horizontal

            # 3. Calculate horizontal distance from bottom to top block
            horizontal_dist = torch.norm(block_poses[-1][..., :2] - block_poses[0][..., :2], dim=-1)

            # Success: bottom block near marker and all blocks stacked
            success = (marker_dist < marker_threshold) & stacked

            return {
                "success": success,
                "horizontal_dist": horizontal_dist,
                "marker_dist": marker_dist,
                "stacked": stacked,
            }
        

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        # Collect block and marker poses for all environments
        num_envs = self.num_envs
        n_block = self.n_block

        # Each block.pose.p is (num_envs, 3), so stack to (num_envs, n_block, 3)
        block_pos = torch.stack([block.pose.p for block in self.blocks], dim=1)  # (num_envs, n_block, 3)
        block_pose = torch.stack([block.pose.raw_pose for block in self.blocks], dim=1)  # (num_envs, n_block, 7)

        # Flatten block_pos and block_pose for compatibility
        block_pos_flat = block_pos.reshape(num_envs, n_block * 3)  # (num_envs, n_block*3)
        block_pose_flat = block_pose.reshape(num_envs, n_block * 7)  # (num_envs, n_block*7)

        marker_pos = self.marker.pose.p  # (num_envs, 3)
        marker_pose = self.marker.pose.raw_pose  # (num_envs, 7)

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            marker_pos=marker_pos,                      # (num_envs, 3)
            block_pos=block_pos_flat,                   # (num_envs, n_block*3)
            block_pose=block_pose_flat,                 # (num_envs, n_block*7)
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Reward function for the task

        """
        with torch.device(self.device):
            # Initialize reward as a 1D tensor with shape (self.num_envs,)
            reward = torch.zeros(self.num_envs, device=self.device)

            return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Normalize the dense reward"""
        max_reward = (
            20.0  # Maximum possible reward (success + all intermediate rewards)
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

