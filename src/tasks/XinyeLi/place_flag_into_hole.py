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
from scipy.spatial.transform import Rotation as R
from mani_skill.utils import common, sapien_utils
from mani_skill.envs.scene import ManiSkillScene
import random

def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder

@register_env("PlaceFlag-v2", max_episode_steps=100)
class PlaceFlagEnv(BaseEnv):
    """
    Task: Place a flag into a vertical hole at the center of a box on the table.
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    def __init__(self, *args, robot_uids="panda", num_envs=1, reconfiguration_freq=None, **kwargs):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(*args, robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        colors = [[0.5, 0.2, 0.8, 1.0], [0.2, 0.5, 0.8, 1.0], [0.8, 0.2, 0.5, 1.0], [0.2, 0.8, 0.5, 1.0], [0.8, 0.5, 0.2, 1.0]]

        # Create the box with a hole
        builder = _build_box_with_hole(self.scene, inner_radius=0.015, outer_radius=0.1, depth=0.05)
        builder.initial_pose = sapien.Pose(p=[-0.1, 0, 0.05], q=[0.7071, 0, 0.7071, 0])
        self.box_with_hole = builder.build_static(name="box_with_hole")

        # Create the flag
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(radius=0.01, half_length=0.1)
        builder.add_cylinder_visual(radius=0.01, half_length=0.1, material=sapien.render.RenderMaterial(base_color=colors[1]))
        builder.add_box_collision(pose=sapien.Pose(p=[0.08, 0.03, 0.005], q=[1, 0, 0, 0]),half_size=(0.02,0.03,0.005))
        builder.add_box_visual(pose=sapien.Pose(p=[0.08, 0.03, 0.005], q=[1, 0, 0, 0]),half_size=(0.02,0.03,0.005),material=sapien.render.RenderMaterial(base_color=colors[2]))
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.005], q=[1, 0, 0, 0])
        self.flag = builder.build(name="flag")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            # Set the initial position of the flag
            flag_pos = torch.zeros((b, 3))
            flag_pos[..., 0] = torch.rand((b,)) * 0.4-0.1
            flag_pos[..., 1] = torch.rand((b,)) * 0.4-0.1
            flag_pos[..., 2] = 0.005
            flag_pose = Pose.create_from_pq(p=flag_pos, q=[1, 0, 0, 0])
            self.flag.set_pose(flag_pose)
            self.flag.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.flag.set_angular_velocity(torch.zeros((b, 3), device=self.device))
            goal_pos = torch.tensor([-0.1, 0, 0.1], device=self.device)  # Center of the box
            goal_pose = Pose.create_from_pq(p=goal_pos, q=[0.7071, 0, -0.7071, 0])
            self.goal_positions = goal_pose


    def evaluate(self):
        """Determine success/failure of the task"""
        with torch.device(self.device):
            flag_position = self.flag.pose.p  # Shape: [b, 3]
            goal_position = self.goal_positions.p  # Center of the box

            # Check if the flag is at the goal position (within threshold)
            at_goal = torch.linalg.norm(flag_position[..., :2] - goal_position[..., :2], dim=-1) < 0.012
            at_goal_z = torch.abs(flag_position[..., 2] - goal_position[..., 2]) < 0.01
            
            # Check if the flag is upright
            # The initial direction of the flag is along the x-axis
            initial_vector = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            # Target direction is the vertical direction (z-axis)
            target_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            
            # We treat the flag pole as a line segment without thickness, only caring about its orientation
            # For a line segment, only the angle with vertical matters, not rotation around its own axis
            # This simplifies the problem and eliminates equivalent rotations that would exist in quaternion space  
            
            # Batch compute the rotated direction
            batch_size = self.num_envs
            rotated_vectors = torch.zeros((batch_size, 3), device=self.device)
            for i in range(batch_size):
                # Convert quaternion to rotation matrix
                q = self.flag.pose.q[i].cpu().numpy()
                q_norm = q / np.linalg.norm(q)
                w, x, y, z = q_norm
                R = np.array([
                    [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
                ])
                rotated_vector = R.dot(initial_vector.cpu().numpy())
                rotated_vectors[i] = torch.tensor(rotated_vector, device=self.device)
            
            # Compute the angle between the rotated vector and the target direction
            rotated_vectors_norm = rotated_vectors / torch.norm(rotated_vectors, dim=1, keepdim=True)
            cos_angles = torch.sum(rotated_vectors_norm * target_direction, dim=1)
            # Avoid numerical issues with acos
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            # Transform to degrees
            angles = torch.acos(cos_angles) * 180.0 / torch.pi
            
            # Define the upright condition: angle with vertical direction < 15 degrees
            upright = angles < 15.0
            
            success = at_goal & at_goal_z & upright
            distance = torch.norm(flag_position[..., :2] - goal_position[..., :2], dim=-1)
            z_distance = torch.abs(flag_position[..., 2] - goal_position[..., 2])
            
            final_eval = {
                "success": success,
                "at_goal": at_goal,
                "at_goal_z": at_goal_z,
                "upright": upright,
                "angle_with_vertical": angles,  # Angles with vertical direction
                "now_flag_q": self.flag.pose.q,
                "goal_q": self.goal_positions.q,
                "distance": distance,
                "z_distance": z_distance,
                "rotated_direction": rotated_vectors  # Vectors after rotation
            }
            return final_eval

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pose=self.goal_positions.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            # Add ground truth information if using state observations
            obs["flag_pose"] = self.flag.pose.raw_pose
            obs["flag_vel"] = self.flag.linear_velocity

        return obs

    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)

            # Success reward
            success = info["success"]
            reward = torch.where(success, reward + 5.0, reward)  # Reduced success reward to make process rewards more meaningful

            # Stage-based rewards - progressive task completion
            flag_position = self.flag.pose.p
            goal_position = self.goal_positions.p
            
            # 1. Horizontal position proximity reward (xy plane)
            xy_distance = torch.linalg.norm(flag_position[..., :2] - goal_position[..., :2], dim=-1)
            position_reward = torch.exp(-3.0 * xy_distance) * 1.0  # Increased weight, reduced decay rate
            reward += position_reward
            
            # 2. Height proximity reward (z-axis)
            z_distance = torch.abs(flag_position[..., 2] - goal_position[..., 2])
            z_reward = torch.exp(-5.0 * z_distance) * 0.8  # Increased weight, moderate decay
            reward += z_reward
            
            # 3. Flag pole vertical alignment reward
            angles = info["angle_with_vertical"]  # Angle with vertical direction (degrees)
            # Angle reward
            # Provides significant positive reward within 30 degrees, but rapidly diminishes beyond 45 degrees
            angle_norm = angles / 90.0  # Normalize to [0,1] range
            upright_reward = torch.exp(-5.0 * angle_norm) * 0.8
            reward += upright_reward
            
            # 4. Combination completion rewards - additional rewards when multiple conditions are met simultaneously
            near_position = xy_distance < 0.05
            near_height = z_distance < 0.03
            near_upright = angles < 20.0
            
            # Position and height both close - extra reward
            position_and_height = near_position & near_height
            reward = torch.where(position_and_height, reward + 0.2, reward)
            
            # Position close and maintaining vertical orientation - extra reward
            position_and_upright = near_position & near_upright
            reward = torch.where(position_and_upright, reward + 0.8, reward)
            
            # Almost complete but not yet successful - substantial reward
            almost_there = near_position & near_height & near_upright & (~success)
            reward = torch.where(almost_there, reward + 1.0, reward)
            
            return reward

    

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward to [0,1] range"""
        # New maximum possible reward: success(5.0) + position(1.0) + height(0.8) + angle(0.8) + combination rewards(0.2+0.8)
        max_reward = 8.6
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

