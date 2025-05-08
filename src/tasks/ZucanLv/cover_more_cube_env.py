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


CUBE_SIZE = [0.02, 0.02, 0.02] # half-sizes for x, y, z (m)
PLANE_RADIUS = 0.08
PLANE_HEIGHT = 0.001


@register_env("CoverMoreCube-v1", max_episode_steps=100)
class CoverMoreCubeEnv(BaseEnv):
    """
    Task: Score points based on how many and which cubes the plane covers. 
          Different cubes carry different point values, and the plane may cover multiple cubes at once. 
          The goal is to maximize your total score.
    """

    """
    Name    Color    Random_Position(center's xy)     Score
    Cube0   green    [0.02,0.08]*[0.02,0.08]          4
    Cube1   yellow   [0.02,0.08]*[-0.08,-0.02]        3
    Cube2   red      [-0.08,-0.02]*[-0.04,0.04]       2  

    Other rewards:
    If plane's z is between (2*CUBE_SIZE[2]-0.01, 2*CUBE_SIZE[2]+0.01), score 1. (It is called plane_suspend in code)
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):

        # Set reconfiguration frequency - for single env, reconfigure every time
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.cubes = []
        for i in range(3):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=CUBE_SIZE)
            if i == 0:
                color = [0 ,1, 0, 1]    # green
            elif i == 1:
                color = [1, 1, 0, 1]    # yellow
            elif i == 2:
                color = [1, 0, 0, 1]    # red
            builder.add_box_visual(
                half_size=CUBE_SIZE,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            builder.initial_pose = sapien.Pose(
                p=[0, 0, CUBE_SIZE[2] + 0.5*(i+1)], q=[1, 0, 0, 0]
            )
            cube = builder.build_static(name=f"cube_{i}")
            self.cubes.append(cube)
        
        # Add a plane
        plane_builder = self.scene.create_actor_builder()
        plane_builder.add_cylinder_collision(radius=PLANE_RADIUS, half_length=PLANE_HEIGHT)
        plane_builder.add_cylinder_visual(radius=PLANE_RADIUS, half_length=PLANE_HEIGHT, material=sapien.render.RenderMaterial(base_color=[1,1,1,0.5])) 
        plane_builder.initial_pose = sapien.Pose(p=[0, 0, PLANE_HEIGHT + 0.1], q=[0.7071, 0, 0.7071, 0]) 
        self.plane = plane_builder.build(name="plane")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            # Randomize the position of the cubes
            for i, cube in enumerate(self.cubes):
                cube_pos = torch.zeros((b, 3))
                if i == 0:
                    cube_pos[..., 0] = torch.rand((b,)) * 0.06 + 0.02
                    cube_pos[..., 1] = torch.rand((b,)) * 0.06 + 0.02
                    cube_pos[..., 2] = CUBE_SIZE[2]
                elif i == 1:
                    cube_pos[..., 0] = torch.rand((b,)) * 0.06 + 0.02
                    cube_pos[..., 1] = torch.rand((b,)) * 0.06 - 0.08
                    cube_pos[..., 2] = CUBE_SIZE[2]
                elif i == 2:
                    cube_pos[..., 0] = torch.rand((b,)) * 0.06 - 0.08
                    cube_pos[..., 1] = torch.rand((b,)) * 0.08 - 0.04
                    cube_pos[..., 2] = CUBE_SIZE[2]
                cube_pose = Pose.create_from_pq(p=cube_pos, q=[1, 0, 0, 0])
                cube.set_pose(cube_pose)

            plane_pos = torch.zeros((b,3))
            plane_pos[..., 0] = torch.rand((b,)) * 0.1 - 0.3
            plane_pos[..., 1] = torch.rand((b,)) * 0.1 - 0.3
            plane_pos[..., 2] = PLANE_HEIGHT
            plane_pose = Pose.create_from_pq(p=plane_pos, q=[0.7071, 0, 0.7071, 0])
            self.plane.set_pose(plane_pose)
            self.plane.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.plane.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        """Determine which cubes are covered by the plane and calculate the score"""
        with torch.device(self.device):
            cube_positions = torch.stack(
                [cube.pose.p for cube in self.cubes]
            )  # Shape: [n_cards, b, 3]

            plane_position = self.plane.pose.p

            # Calculate the horizontal distance (xy plane) between each cube and the plane
            # Only consider x and y coordinates for distance calculation
            distances_xy = torch.norm(
                cube_positions[:, :, :2] - plane_position[:, :2].unsqueeze(0),
                dim=2
            )  # Shape: [n_cubes, b]

            # Check if cubes are covered (distance < plane radius)
            is_covered = distances_xy < PLANE_RADIUS  # Shape: [n_cubes, b]

            # Convert boolean tensors to integers (0 or 1)
            cube0_iscover = is_covered[0, :].int()
            cube1_iscover = is_covered[1, :].int()
            cube2_iscover = is_covered[2, :].int()
            
            # Check if the plane is suspended at the right height
            # plane_suspend is 1 if plane's z is between (2*CUBE_SIZE[2]-0.01, 2*CUBE_SIZE[2]+0.01)
            z_min = 2 * CUBE_SIZE[2] - 0.01
            z_max = 2 * CUBE_SIZE[2] + 0.01
            plane_suspend = ((plane_position[:, 2] > z_min) & (plane_position[:, 2] < z_max)).int()

            return {
                "cube0_iscover": cube0_iscover,
                "cube1_iscover": cube1_iscover,
                "cube2_iscover": cube2_iscover,
                "plane_suspend": plane_suspend
            }
    
    def _get_bos_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            plane_pos=self.plane.pose.p,
        )
        return obs
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward for the task"""
        with torch.device(self.device):
            cube0_iscover = info["cube0_iscover"]
            cube1_iscover = info["cube1_iscover"]
            cube2_iscover = info["cube2_iscover"]
            plane_suspend = info["plane_suspend"]

            # Calculate the score based on which cubes are covered
            total_score = 4 * cube0_iscover + 3 * cube1_iscover + 2 * cube2_iscover + plane_suspend

            return total_score
        
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Normalize the dense reward"""
        max_reward = (
            10.0  # Maximum possible reward (success + all intermediate rewards)
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

