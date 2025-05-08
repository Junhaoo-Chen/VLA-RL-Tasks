from typing import Dict, Any
import numpy as np
import gymnasium as gym
import torch
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import Panda
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.sapien_utils import look_at

@register_env("ColorSortEnv-v1", max_episode_steps=100)
class ColorSortEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args,  robot_uids="panda", **kwargs):

        self.types = ["capsule"]*5+ ["box"]*5+ ["cylinder"]*5
        self.colors = {
            "capsule": [0.8, 0.2, 0.2, 1], 
            "box": [0.2, 0.8, 0.2, 1],  
            "cylinder": [0.2, 0.2, 0.8, 1],  
        }
        self.bin_locations = {
            "capsule": [0.3, -0.2, 0],
            "box": [0.3, 0.0, 0],
            "cylinder": [0.3, 0.2, 0],
        }
        super().__init__(*args,  robot_uids=robot_uids, **kwargs)


    def _load_scene(self, options: dict):

        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.objects = []
        for i, type in enumerate(self.types):
            builder = self.scene.create_actor_builder()
            builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
            if type == "capsule":
                builder.add_capsule_visual(radius=0.01, half_length=0.02, material=sapien.render.RenderMaterial(base_color=self.colors[type]))
                builder.add_capsule_collision(radius=0.01, half_length=0.02)
            elif type == "box":
                builder.add_box_visual(half_size=[0.01, 0.01, 0.02],material=sapien.render.RenderMaterial(base_color=self.colors[type]))
                builder.add_box_collision(half_size=[0.01, 0.01, 0.02])
            elif type == "cylinder":
                builder.add_cylinder_visual(radius=0.01, half_length=0.005,material=sapien.render.RenderMaterial(base_color=self.colors[type]))
                builder.add_cylinder_collision(radius=0.01, half_length=0.005)
            self.objects.append(builder.build(name=f"{type}_{i}")) 

        self.bins = {}
        for type, loc in self.bin_locations.items():
            builder = self.scene.create_actor_builder()
            bin_material = sapien.render.RenderMaterial(base_color=self.colors[type])
            builder.add_box_visual(half_size=[0.05, 0.05, 0.02],material=bin_material)
            builder.add_box_collision(half_size=[0.05, 0.05, 0.02])
            self.bins[type] = builder.build_static(name=f"{type}_bin")
            self.bins[type].set_pose(sapien.Pose(loc))


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        spawn_area_min = torch.tensor([-0.4, -0.3, 0.02])  # x: -0.4 to -0.2, y: -0.3 to 0.3
        spawn_area_max = torch.tensor([-0.2, 0.3, 0.02])
        
        num_objects = len(self.objects)
        random_positions = torch.rand((num_objects, 3)) * (spawn_area_max - spawn_area_min) + spawn_area_min
        
        random_rotations = torch.rand(num_objects) * 2 * np.pi  
        
        for i, med in enumerate(self.objects):
            med.set_pose(Pose.create_from_pq(
                p=random_positions[i],
                q=euler2quat(np.pi / 2, np.pi / 2, random_rotations[i]) 
            ))

    def evaluate(self):

        success = torch.ones(len(self.objects), dtype=torch.bool)
        for object in self.objects:
            type = object.name.split("_")[0]
            bin_center = torch.tensor(self.bin_locations[type])
            dist = torch.norm(object.pose.p[:, :2] - bin_center[:2], dim=1)
            success &= (dist < 0.05)  
        return {"success": success}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        reward = torch.zeros(len(self.objects))
        for object in self.objects:
            type = object.name.split("_")[0]
            bin_center = torch.tensor(self.bin_locations[type])
            dist = torch.norm(object.pose.p[:, :2] - bin_center[:2], dim=1)
            reward += 1 - torch.tanh(5 * dist)  
        reward[info["success"]] = 15.0  
        return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 15.0
        return self.compute_dense_reward(obs, action, info) / max_reward

    @property
    def _default_sensor_configs(self):
 
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )


        side_camera = CameraConfig(
            "side_camera",
            pose=look_at(eye=[0.5, 0, 0.5], target=[0, 0, 0.2]),
            width=128,
            height=100,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
        return [top_camera, side_camera]

    @property
    def _default_human_render_camera_configs(self):

        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[0.6, 0.6, 0.6], target=[0, 0, 0.1]),
            width=512,
            height=256,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )