from typing import Dict, Union, Any
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.sapien_utils import look_at
from transforms3d.euler import euler2quat

@register_env("FiveInARowEnv-v0", max_episode_steps=50)
class FiveInARowEnv(BaseEnv):
    """
    Task description:  You are playing a Five-in-a-Row game and it's now your turn to place a black chess piece. Win over your opponent! 
    """
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        self.grid_size = 6
        self.cell_size = 0.1
        self.canvas_thickness = 0.02
        self.canvas_half_size = [self.grid_size * self.cell_size / 2] * 2
        self.canvas_center = [0.1, 0, self.canvas_thickness / 2]

        # create a board
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[*self.canvas_half_size, self.canvas_thickness / 2],
            material=sapien.render.RenderMaterial(base_color=[1.0, 0.9, 0.6, 1], transmission=1.0)
        )
        builder.add_box_collision(
            half_size=[*self.canvas_half_size, self.canvas_thickness / 2]
        )
        builder.initial_pose = sapien.Pose(p=self.canvas_center)
        self.canvas = builder.build_static(name="canvas")

        self._draw_grid_lines()

        self._place_stones()

        self._create_stone_boxes()

    def _draw_grid_lines(self):
        lines = []
        start_x = self.canvas_center[0] - self.canvas_half_size[0]
        start_y = self.canvas_center[1] - self.canvas_half_size[1]
        for i in range(self.grid_size + 1):
            # vertical lines
            x = start_x + i * self.cell_size
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[0.001, self.canvas_half_size[1], 0.001],
                material=sapien.render.RenderMaterial(base_color=[0, 0, 0, 1])
            )
            builder.initial_pose = sapien.Pose(p=[x, self.canvas_center[1], self.canvas_center[2] + 0.011])
            lines.append(builder.build_static(name=f"vline_{i}"))

            # horizontal lines
            y = start_y + i * self.cell_size
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                half_size=[self.canvas_half_size[0], 0.001, 0.001],
                material=sapien.render.RenderMaterial(base_color=[0, 0, 0, 1])
            )
            builder.initial_pose = sapien.Pose(p=[self.canvas_center[0], y, self.canvas_center[2] + 0.011])
            lines.append(builder.build_static(name=f"hline_{i}"))

        self.grid_lines = lines

    def _place_stones(self):
        # 0 = empty, 1 = black, 2 = white
        layout = np.array([
            [1, 1, 0, 2, 0, 0],
            [2, 0, 1, 1, 0, 0],
            [0, 2, 1, 1, 0, 0],
            [0, 2, 2, 1, 0, 0],
            [0, 2, 2, 2, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ])
        self.stones = []
        radius = 0.015
        height = 0.02

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = layout[i, j]
                if val == 0:
                    continue
                color = [0, 0, 0, 1] if val == 1 else [1, 1, 1, 1]

                builder = self.scene.create_actor_builder()
                builder.add_cylinder_visual(
                    radius=radius,
                    half_length=height / 2,
                    material=sapien.render.RenderMaterial(base_color=color)
                )
                builder.add_cylinder_collision(
                    radius=radius,
                    half_length=height / 2,
                    density=500
                )

                x = self.canvas_center[0] - self.canvas_half_size[0] + (j + 0.5) * self.cell_size
                y = self.canvas_center[1] - self.canvas_half_size[1] + (i + 0.5) * self.cell_size
                z = self.canvas_center[2] + self.canvas_thickness / 2 + height / 2

                builder.initial_pose = sapien.Pose(p=[x, y, z], q=euler2quat(0, np.pi / 2, 0))
                self.stones.append(builder.build_static(name=f"stone_{i}_{j}"))

    
    def _create_stone_boxes(self):
        wall_thickness = 0.005
        box_inner_size = [0.1, 0.1]
        box_height = 0.1

        offset = self.canvas_half_size[0] + box_inner_size[0] / 2 + 0.05
        box_z = self.canvas_center[2] + self.canvas_thickness / 2 + box_height / 2

        # on the two sides of board
        black_box_center = [self.canvas_center[0], 
                            self.canvas_center[1] - offset, 
                            box_z]
        white_box_center = [self.canvas_center[0], 
                            self.canvas_center[1] + offset, 
                            box_z]

        # black & white stone box
        self.black_box_parts = self._build_open_box(black_box_center, box_inner_size, box_height, wall_thickness, [0.2, 0.2, 0.2, 1], "black")
        self.white_box_parts = self._build_open_box(white_box_center, box_inner_size, box_height, wall_thickness, [0.8, 0.8, 0.8, 1], "white")

        # place the stones (in box)
        self._place_stones_in_box(black_box_center, color=[0, 0, 0, 1], count=2, prefix="black")
        self._place_stones_in_box(white_box_center, color=[1, 1, 1, 1], count=3, prefix="white")
    
    def _build_open_box(self, center, inner_size, height, thickness, color, prefix):
        parts = []
        dx, dy = inner_size[0] / 2 + thickness / 2, inner_size[1] / 2 + thickness / 2

        # four walls
        wall_positions = [
            [center[0] - dx, center[1], center[2]],  # left
            [center[0] + dx, center[1], center[2]],  # right
            [center[0], center[1] - dy, center[2]],  # up
            [center[0], center[1] + dy, center[2]],  # down
        ]
        wall_sizes = [
            [thickness / 2, inner_size[1] / 2 + thickness, height / 2],
            [thickness / 2, inner_size[1] / 2 + thickness, height / 2],
            [inner_size[0] / 2, thickness / 2, height / 2],
            [inner_size[0] / 2, thickness / 2, height / 2],
        ]

        for i in range(4):
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=wall_sizes[i], material=sapien.render.RenderMaterial(base_color=color))
            builder.add_box_collision(half_size=wall_sizes[i])
            builder.initial_pose = sapien.Pose(p=wall_positions[i])
            parts.append(builder.build_static(name=f"{prefix}_box_wall_{i}"))

        return parts

    
    def _place_stones_in_box(self, box_center, color, count, prefix):
        radius = 0.015
        height = 0.02
        stack_spacing = height + 0.005
        base_z = box_center[2] + 0.05

        for i in range(count):
            x, y = box_center[0], box_center[1]
            z = base_z + i * stack_spacing + height / 2

            builder = self.scene.create_actor_builder()
            builder.add_cylinder_visual(
                radius=radius,
                half_length=height / 2,
                material=sapien.render.RenderMaterial(base_color=color)
            )
            builder.add_cylinder_collision(
                radius=radius,
                half_length=height / 2,
                density=500         # let gravity do the rest
            )
            builder.initial_pose = sapien.Pose(p=[x, y, z], q=euler2quat(0, np.pi / 2, 0))
            builder.build(name=f"{prefix}_box_stone_{i}")


    def _initialize_episode(self, env_idx, options):
        self.table_scene.initialize(env_idx)

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=look_at(eye=[0.5, 0.5, 0.5], target=[0.3, 0.0, 0.0]),
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100
        )

    def evaluate(self):
        # Calculate the coordinates of (3, 4)
        target_x = self.canvas_center[0] - self.canvas_half_size[0] + (4 + 0.5) * self.cell_size
        target_y = self.canvas_center[1] - self.canvas_half_size[1] + (3 + 0.5) * self.cell_size
        target_pos = torch.tensor([target_x, target_y], device=self.device)

        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        min_dis = torch.full((self.num_envs,), float('inf'), device=self.device)

        for actor in self.scene.get_all_actors():
            if actor.name.startswith("black_box_stone"):
                pos = actor.pose.p  # Shape: [num_envs, 3]
                dist = torch.norm(pos[:, :2] - target_pos, dim=1)  # Shape: [num_envs]
                success |= dist < 0.03
                min_dis = torch.minimum(min_dis, dist)

        return {
            "success": success,
            "min_distance": min_dis   # Shape: [num_envs]
        }
        
       
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 10.0
        min_dis = info.get("min_distance", torch.full((self.num_envs,), float('inf'), device=self.device))
        success = info.get("success", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

        reward = torch.zeros(self.num_envs, device=self.device)
        reward[success] = max_reward
        shaping = max_reward * (1 - torch.tanh(5 * min_dis))
        reward[~success] = shaping[~success]

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 10.0
        dense_reward = self.compute_dense_reward(obs, action, info)
        return dense_reward / max_reward
