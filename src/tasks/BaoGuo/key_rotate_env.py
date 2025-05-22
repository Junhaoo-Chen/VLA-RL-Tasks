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


KEY_SIZE = [0.04, 0.04, 0.2]  # Size of the key
HOLLOW_BOX_SIZE = [0.2, 0.2, 0.1, 0.06]  # Size of the hollow box (Outer width, depth, height, wall thickness)


@register_env("KeyRotate-v1", max_episode_steps=100)
class KeyRotateEnv(BaseEnv):
    """KeyRotate
        Task: insert a key into another object and rotate it
        Goal: focus on the
        
        Robot: Panda
        Object: Retangle-liked object as key , a hollow object as the keyhold
        Scene: Tabletop
        Action: joint position / trajectory
    """
    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        # Ensure the scene is passed correctly to TableSceneBuilder
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        material = sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
        self.hollow_box_actor = create_hollow_box(
            scene=self.scene,
            center=[0, 0, 0],
            outer_size=HOLLOW_BOX_SIZE[:2],  # Outer width and depth
            wall_thickness=HOLLOW_BOX_SIZE[3],    # Thickness of the walls
            height=HOLLOW_BOX_SIZE[2],             # Height of the walls
            material=material,
        )
        # Create a key
        key_builder = self.scene.create_actor_builder()
        key_builder.add_box_collision(
            half_size=KEY_SIZE,
            pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        key_builder.add_box_visual(
            half_size=KEY_SIZE,
            pose=sapien.Pose(p=[0, 0, 0.1]),
            material=sapien.render.RenderMaterial(
                base_color=[1.0, 0.2, 0.4, 1],
            ),
        )
        self.key_actor = key_builder.build(name="key")
        self.key_actor.set_pose(sapien.Pose(p=[0.2, 0.2, KEY_SIZE[2]]))
        
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            box_pos_init = torch.zeros((b, 3))
            box_pos_init[:, 0] = torch.rand(b) * 0.3 + 0.2
            box_pos_init[:, 1] = torch.rand(b) * 0.3 + 0.2
            box_pos_init[:, 2] = HOLLOW_BOX_SIZE[2]
            
            key_pos_init = torch.zeros((b, 3))
            key_pos_init[:, 0] = torch.rand(b) * 0.3 - 0.2
            key_pos_init[:, 1] = torch.rand(b) * 0.3 - 0.2
            key_pos_init[:, 2] = KEY_SIZE[2]
            
            # Set the initial positions of the hollow box and key
            box_pose = Pose.create_from_pq(p = box_pos_init, q= [0, 0, 0, 1])
            self.hollow_box_actor.set_pose(box_pose)
            
            key_pose = Pose.create_from_pq(p = key_pos_init, q= [0, 0, 0, 1])
            self.key_actor.set_pose(key_pose)
            
            # Set Goal
            self.goal_pos = box_pos_init.clone()
            self.goal_pose = torch.zeros((b, 4))
            self.goal_pose[:, 2] = np.sin(np.pi / 4)  # sin(90° / 2)
            self.goal_pose[:, 3] = np.cos(np.pi / 4)  # cos(90° / 2)
            

    def evaluate(self):
        """Determine success/failure of the task"""
        with torch.device(self.device):
            # Check if the key is inside the hollow box
            key_pos = self.key_actor.pose.p
            box_pos = self.hollow_box_actor.pose.p
            inside_box = torch.tensor(
                torch.allclose(key_pos, box_pos, atol=0.1), device=self.device
            ).unsqueeze(0)  
            
            # Check if the key is rotated correctly
            box_rotation = self.hollow_box_actor.pose.q
            rotated_correctly = torch.tensor(
                torch.allclose(self.goal_pose, box_rotation, atol=0.1), device=self.device
            ).unsqueeze(0)  
            
            # Compute success
            success = torch.logical_and(inside_box, rotated_correctly)
            
            angle_to_goal = torch.acos(
                torch.clamp(
                    torch.sum(self.goal_pose * box_rotation, dim=-1), -1.0, 1.0
                )
            ).unsqueeze(0)  
            
            return {
                "success": success,
                "inside_box": inside_box,
                "rotated_correctly": rotated_correctly,
                "angle_to_goal": angle_to_goal,
            }
    
        

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_pos,
            goal_rot=self.goal_pose,
            key_pos=self.key_actor.pose.p,
            key_rot=self.key_actor.pose.q,
            box_pos=self.hollow_box_actor.pose.p,
            box_rot=self.hollow_box_actor.pose.q,
            inside_box=torch.tensor([self.evaluate()["inside_box"]], device=self.device),
            rotated_correctly=torch.tensor([self.evaluate()["rotated_correctly"]], device=self.device),
        )
        
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Reward function for the task
            The reward is based on the position and pose of the key
            For each sub-goal achieved, it will receive a reward respectively
            for incomplete sub-goal like rotate to goal pose, it will receive a reward inversely proportional to the distance to the goal
            
        """
        with torch.device(self.device):
            # Initialize reward as a 1D tensor with shape (self.num_envs,)
            reward = torch.zeros(self.num_envs, device=self.device)

            # Check if the key is inside the hollow box
            inside_box = info["inside_box"]
            rotated_correctly = info["rotated_correctly"]
            # Reward for being inside the box
            if inside_box:
                reward += 7.0
            # Reward for being rotated correctly
            if rotated_correctly:
                reward += 3.0
            # Reward for being close to the goal position
            angle_to_goal = info["angle_to_goal"]
            reward += 5.0 / (1.0 + angle_to_goal.view(-1))

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
        
        
def create_hollow_box(scene, center, outer_size, wall_thickness, height, material):
    """
    Create a hollow box using thin walls.

    Args:
        scene: The SAPIEN scene.
        center: The center position of the hollow box.
        outer_size: The size of the outer box [width, depth].
        wall_thickness: The thickness of the walls.
        height: The height of the walls.
        material: The visual material for the walls.
    """
    builder = scene.create_actor_builder()

    # Outer dimensions
    width, depth = outer_size

    # Add four walls
    # Left wall
    builder.add_box_collision(
        half_size=[wall_thickness / 2, depth / 2, height / 2],
        pose=sapien.Pose(p=[-width / 2 + wall_thickness / 2, 0, height / 2]),
    )
    builder.add_box_visual(
        half_size=[wall_thickness / 2, depth / 2, height / 2],
        pose=sapien.Pose(p=[-width / 2 + wall_thickness / 2, 0, height / 2]),
        material=material,
    )

    # Right wall
    builder.add_box_collision(
        half_size=[wall_thickness / 2, depth / 2, height / 2],
        pose=sapien.Pose(p=[width / 2 - wall_thickness / 2, 0, height / 2]),
    )
    builder.add_box_visual(
        half_size=[wall_thickness / 2, depth / 2, height / 2],
        pose=sapien.Pose(p=[width / 2 - wall_thickness / 2, 0, height / 2]),
        material=material,
    )

    # Front wall
    builder.add_box_collision(
        half_size=[width / 2, wall_thickness / 2, height / 2],
        pose=sapien.Pose(p=[0, -depth / 2 + wall_thickness / 2, height / 2]),
    )
    builder.add_box_visual(
        half_size=[width / 2, wall_thickness / 2, height / 2],
        pose=sapien.Pose(p=[0, -depth / 2 + wall_thickness / 2, height / 2]),
        material=material,
    )

    # Back wall
    builder.add_box_collision(
        half_size=[width / 2, wall_thickness / 2, height / 2],
        pose=sapien.Pose(p=[0, depth / 2 - wall_thickness / 2, height / 2]),
    )
    builder.add_box_visual(
        half_size=[width / 2, wall_thickness / 2, height / 2],
        pose=sapien.Pose(p=[0, depth / 2 - wall_thickness / 2, height / 2]),
        material=material,
    )

    # Build the hollow box
    hollow_box = builder.build_static(name="hollow_box")
    hollow_box.set_pose(sapien.Pose(center))
    return hollow_box