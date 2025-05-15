
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

CARD_SIZE = [0.05, 0.08, 0.003]  # half-sizes for x, y, z (mm)
GRID_SIZE_X = 0.12  # spacing between cards in x direction
GRID_SIZE_Y = 0.18  # spacing between cards in y direction
GRID_OFFSET_X = -0.18  # center of grid in x
GRID_OFFSET_Y = -0.24  # center of grid in y

@register_env("FindTreasure-v1", max_episode_steps=200)
class FindTreasureEnv(BaseEnv):
    """
    Task: Identify and move the correct green-marked card to the target place from a 3x4 grid of cards.
    Each card has a colored marker underneath (green=target, yellow=near target, red=no target nearby).
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", num_envs=1,
        reconfiguration_freq=None, **kwargs):
        
        # Set reconfiguration frequency - for single env, reconfigure every time
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(*args, robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)
    
    def _load_agent(self, options):
        return super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        # Create all possible markers (static, cannot be moved) with different colors
        # We'll create 12 markers for each color (green, yellow, red) and hide them initially
        self.green_markers = []
        self.yellow_markers = []
        self.red_markers = []
        
        # Initial position under the table (hidden)
        hidden_pos = [0, 0, -0.5]
        
        # Create green markers (for target cards)
        for i in range(12):
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=[CARD_SIZE[0], CARD_SIZE[1], 0.001],
                material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]))
            builder.initial_pose = sapien.Pose(p=hidden_pos, q=[1, 0, 0, 0])
            marker = builder.build_static(name=f"green_marker_{i}")
            self.green_markers.append(marker)
        
        # Create yellow markers (for cards near target)
        for i in range(12):
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=[CARD_SIZE[0], CARD_SIZE[1], 0.001],
                material=sapien.render.RenderMaterial(base_color=[1, 1, 0, 1]))
            builder.initial_pose = sapien.Pose(p=hidden_pos, q=[1, 0, 0, 0])
            marker = builder.build_static(name=f"yellow_marker_{i}")
            self.yellow_markers.append(marker)
        
        # Create red markers (for cards far from target)
        for i in range(12):
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=[CARD_SIZE[0], CARD_SIZE[1], 0.001],
                material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]))
            builder.initial_pose = sapien.Pose(p=hidden_pos, q=[1, 0, 0, 0])
            marker = builder.build_static(name=f"red_marker_{i}")
            self.red_markers.append(marker)
        
        # Create movable cards (will cover the markers)
        self.cards = []
        for i in range(12):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=CARD_SIZE)
            builder.add_box_visual(half_size=CARD_SIZE, 
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1]))
            builder.initial_pose = sapien.Pose(p=[0, 0, CARD_SIZE[2] + 0.5], q=[1, 0, 0, 0])
            card = builder.build(name=f"card_{i}")
            self.cards.append(card)
        
        # Create target place (visual only)
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[CARD_SIZE[0], CARD_SIZE[1], 0.001],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.5]))
        builder.initial_pose = sapien.Pose(p=[0.2, 0, 0.001], q=[1, 0, 0, 0])
        self.target_marker = builder.build_static(name="target_marker")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        
        with torch.device(self.device):
            b = len(env_idx)
            
            # Randomly select target card position (0-11)
            self.target_idx = torch.randint(0, 12, (b,), device=self.device)
            
            # First hide all markers by moving them under the table
            hidden_pos = torch.tensor([0, 0, -0.5], device=self.device).repeat(b, 1)
            hidden_pose = Pose.create_from_pq(p=hidden_pos, q=[1, 0, 0, 0])
            
            for marker in self.green_markers + self.yellow_markers + self.red_markers:
                marker.set_pose(hidden_pose)
            
            # Calculate grid positions (row, col) for all cards
            grid_positions = torch.tensor([
                (idx // 4, idx % 4) for idx in range(12)
            ], device=self.device)
            
            # Get target position
            target_positions = grid_positions[self.target_idx]
            
            # Position the markers based on their distance to target
            for i in range(b):
                target_pos = target_positions[i]
                for idx in range(12):
                    current_pos = grid_positions[idx]
                    distance = torch.sum(torch.abs(current_pos - target_pos))
                    
                    # Calculate marker position in world coordinates
                    row = idx // 4
                    col = idx % 4
                    marker_pos = torch.tensor([
                        GRID_OFFSET_X + row * GRID_SIZE_X,
                        GRID_OFFSET_Y + col * GRID_SIZE_Y,
                        0.001
                    ], device=self.device)
                    
                    # Select which marker to show based on distance
                    if idx == self.target_idx[i]:
                        # Show green marker for target card
                        self.green_markers[idx].set_pose(
                            Pose.create_from_pq(p=marker_pos, q=[1, 0, 0, 0])
                        )
                    elif distance <= 1:  # adjacent cards
                        # Show yellow marker for nearby cards
                        self.yellow_markers[idx].set_pose(
                            Pose.create_from_pq(p=marker_pos, q=[1, 0, 0, 0])
                        )
                    else:
                        # Show red marker for far cards
                        self.red_markers[idx].set_pose(
                            Pose.create_from_pq(p=marker_pos, q=[1, 0, 0, 0])
                        )
            
            # Position cards above their markers (initially covering them)
            for i, card in enumerate(self.cards):
                row = i // 4
                col = i % 4
                card_pos = torch.tensor([
                    GRID_OFFSET_X + row * GRID_SIZE_X,
                    GRID_OFFSET_Y + col * GRID_SIZE_Y,
                    CARD_SIZE[2]
                ], device=self.device).repeat(b, 1)
                card.set_pose(Pose.create_from_pq(p=card_pos, q=[1, 0, 0, 0]))
                
                # Reset velocities
                card.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                card.set_angular_velocity(torch.zeros((b, 3), device=self.device))
            
            # Set target position (fixed for now)
            target_pos = torch.tensor([0.2, 0, 0.001], device=self.device).repeat(b, 1)
            self.target_marker.set_pose(Pose.create_from_pq(p=target_pos, q=[1, 0, 0, 0]))
            self.target_position = target_pos

    def evaluate(self):
        """Determine success/failure of the task"""
        with torch.device(self.device):
            # Get positions of all cards
            card_positions = torch.stack([card.pose.p for card in self.cards])  # Shape: [12, b, 3]
            
            # Check if target card is at target position
            target_card_pos = card_positions[self.target_idx, torch.arange(self.num_envs)]
            at_target = torch.linalg.norm(
                target_card_pos[:, :2] - self.target_position[:, :2], dim=-1
            ) < (CARD_SIZE[0] + CARD_SIZE[1]) / 2
            
            # Check if other cards are not at target position
            other_cards_at_target = torch.zeros_like(at_target)
            for i in range(12):
                if i != self.target_idx:
                    dist = torch.linalg.norm(
                        card_positions[i, :, :2] - self.target_position[:, :2], dim=-1)
                    other_cards_at_target = torch.logical_or(
                        other_cards_at_target, 
                        dist < (CARD_SIZE[0] + CARD_SIZE[1]) / 2
                    )
            
            return {
                "success": at_target & ~other_cards_at_target,
                "target_at_goal": at_target,
                "wrong_at_goal": other_cards_at_target,
                "target_idx": self.target_idx,
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_pos=self.target_position,
        )
        
        if self.obs_mode_struct.use_state:
            # Add ground truth information if using state observations
            for i, card in enumerate(self.cards):
                obs[f"card_{i}_pose"] = card.pose.raw_pose
                obs[f"card_{i}_vel"] = card.linear_velocity
        
        return obs
        
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)
            
            # Success reward
            success = info["success"]
            reward = torch.where(success, reward + 10.0, reward)
            
            # Reward for having target card close to goal
            target_card_pos = self.cards[self.target_idx].pose.p
            target_dist = torch.linalg.norm(
                target_card_pos[:, :2] - self.target_position[:, :2], dim=-1)
            reward += torch.exp(-5.0 * target_dist) * 2.0
            
            # Penalty for having wrong cards at goal
            wrong_at_goal = info["wrong_at_goal"]
            reward = torch.where(wrong_at_goal, reward - 1.0, reward)
            
            # Reward for getting the TCP close to target card
            tcp_pos = self.agent.tcp.pose.p
            tcp_to_target = torch.linalg.norm(
                target_card_pos - tcp_pos, dim=-1)
            reward += torch.exp(-10.0 * tcp_to_target) * 0.5
            
            return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward"""
        max_reward = 12.5  # Maximum possible reward (success + all intermediate rewards)
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig("top_camera", pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=128, height=128, fov=np.pi/3, near=0.01, far=100)
        
        # Side view camera
        side_camera = CameraConfig("side_camera", pose=look_at(eye=[0.5, 0, 0.5], target=[0, 0, 0.2]),
            width=128, height=128, fov=np.pi/3, near=0.01, far=100)
        return [top_camera, side_camera]
    
    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human viewing"""
        return CameraConfig("render_camera", pose=look_at(eye=[0.6, 0.6, 0.6], target=[0, 0, 0.1]),
            width=512, height=512, fov=np.pi/3, near=0.01, far=100)