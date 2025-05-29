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


BLOCK_SIZE = [0.03, 0.03, 0.1]  # half-sizes for x, y, z (m)
CYLINDER_SIZE = [0.03, 0.1]  # radius, half-height
BOTTLE_SIZE = [0.05, 0.1]  # radius, half-height

@register_env("LiftUp-v1", max_episode_steps=100)
class FLiftUpEnv(BaseEnv):
    """LiftUp
        Task: Stand up all the lying / falling objects, Make them upright
        Goal: focus on the trajectory of the robot on interaction with the objects
        
        Robot: Panda
        Object: cuboid blocks, bottles, cylinders, etc.
        Scene: Tabletop
        Action: joint position / trajectory
    """
    def __init__(self, *args, robot_uids="panda", n_obj=3, **kwargs):
        self.n_obj = n_obj
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with a table and objects
            Random number of objects are placed on the table,
            each with a random pose and type (block, bottle, cylinder, etc.)
            But all of them are lying down, and the goal is to make them upright.
            The total number of objects is determined by `n_obj` parameter.
        """
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # Define object types and their sizes
        obj_types = ["block", "cylinder"]
        obj_sizes = {
            "block": BLOCK_SIZE,      # half-sizes (x, y, z)
            "cylinder": CYLINDER_SIZE,         # radius, half-height
            "bottle": BOTTLE_SIZE,         # radius, half-height
        }
        self.objects = []
        for i in range(self.n_obj):
            obj_type = np.random.choice(obj_types)
            builder = self.scene.create_actor_builder()
            color = np.random.rand(4)
            color[3] = 1.0  # alpha

            if obj_type == "block":
                builder.add_box_collision(half_size=obj_sizes["block"])
                builder.add_box_visual(
                    half_size=obj_sizes["block"],
                    material=sapien.render.RenderMaterial(base_color=color.tolist()),
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
                # Lying down: flat on table, random xy, z = block half-height
                pose = sapien.Pose(
                    p=[np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), obj_sizes["block"][2]],
                    q=[1, 0, 0, 0]
                )
            elif obj_type == "cylinder":
                builder.add_cylinder_collision(radius=obj_sizes["cylinder"][0], half_length=obj_sizes["cylinder"][1])
                builder.add_cylinder_visual(
                    radius=obj_sizes["cylinder"][0], half_length=obj_sizes["cylinder"][1],
                    material=sapien.render.RenderMaterial(base_color=color.tolist()),
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
                # Lying down: cylinder axis along x, so rotate 90deg about y
                pose = sapien.Pose(
                    p=[np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), obj_sizes["cylinder"][0]],
                    q=[np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]  # 90deg about y
                )
            elif obj_type == "bottle":
                builder.add_capsule_collision(radius=obj_sizes["bottle"][0], half_length=obj_sizes["bottle"][1])
                builder.add_capsule_visual(
                    radius=obj_sizes["bottle"][0], half_length=obj_sizes["bottle"][1],
                    material=sapien.render.RenderMaterial(base_color=color.tolist()),
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
                # Lying down: capsule axis along x, so rotate 90deg about y
                pose = sapien.Pose(
                    p=[np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), obj_sizes["bottle"][0]],
                    q=[np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]  # 90deg about y
                )
            obj = builder.build(name=f"{obj_type}_{i}")
            obj.set_pose(pose)
            self.objects.append(obj)
    
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            # For each object, set a random pose for each environment
            for obj in self.objects:
                # Random (x, y) for each env, z depends on object type
                obj_name = obj.name
                if obj_name.startswith("block"):
                    z = BLOCK_SIZE[0]  
                    q = torch.tensor([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0], device=self.device)
                elif obj_name.startswith("cylinder"):
                    z = CYLINDER_SIZE[0]  
                    q = torch.tensor([0, 0, 0, 1], device=self.device)
                elif obj_name.startswith("bottle"):
                    z = BOTTLE_SIZE[0]
                    q = torch.tensor([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0], device=self.device)
                else:
                    z = 0.01
                    q = torch.tensor([1, 0, 0, 0], device=self.device)

                pos = torch.zeros(b, 3, device=self.device)
                pos[..., 0] = torch.rand(b, device=self.device) * 0.5 - 0.2
                pos[..., 1] = torch.rand(b, device=self.device) * 0.5 - 0.2
                pos[..., 2] = z

                pose = Pose.create_from_pq(p=pos, q=q)
                obj.set_pose(pose)
                obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))
            

    def evaluate(self):
        """Determine success/failure of the task
            Description: 
                1. Check if all objects are upright (z-axis aligned with world up)
                2. If all objects are upright, return success
                3. return number of upright objects
        """
        with torch.device(self.device):
            upright_counts = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
            upright_threshold = 0.95  # cosine similarity threshold for "upright"

            for obj in self.objects:
                # Get orientation quaternion (w, x, y, z)
                quat = obj.pose.q  # shape: (num_envs, 4)
                # Compute z-axis in world frame from quaternion
                # z_axis = [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
                w, x, y, z = quat.unbind(-1)
                z_axis_x = 2 * (x * z + w * y)
                z_axis_y = 2 * (y * z - w * x)
                z_axis_z = 1 - 2 * (x * x + y * y)
                z_axis = torch.stack([z_axis_x, z_axis_y, z_axis_z], dim=-1)  # (num_envs, 3)
                upright = z_axis[..., 2] > upright_threshold  # dot([0,0,1], z_axis) = z_axis[2]
                upright_counts += upright.to(torch.int)

            success = upright_counts == len(self.objects)
            return {
                "success": success,
                "upright_counts": upright_counts,
            }
        

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        # Collect block and marker poses for all environments
        num_envs = self.num_envs


        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
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

