import os
import sapien
import torch
import math
import numpy as np
from typing import Dict, Any, Union, List
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.sapien_utils import look_at
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig


@register_env("BalanceBeam-v1", max_episode_steps=400)
class BalanceBeamEnv(BaseEnv):
    """
    **Task Description:**
    Place all the cubes on two sides of the balance beam without causing it to fall down.

    **Instruction:**
    "Put all the cubes on the sides of the balance beam"

    **Randomization:**
    - 10 cubes with fixed sizes: 4 cubes of size 0.03, 2 cubes of size 0.04, 2 cubes of size 0.05, and 2 cubes of size 0.06
    - Cubes are randomly placed near the balance beam

    **Success Condition:**
    - All cubes are placed on the balance beam
    - The balance beam has not fallen down
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # Parameters
    num_cubes = 10
    beam_size = [0.05, 0.4, 0.02]  # width, length, height
    cube_size_range = [0.03, 0.06]  # min and max side length

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # Load balance base
        balance_base_model_file = os.path.join(
            os.path.dirname(__file__), "assets", "balanceBase.obj"
        )
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(balance_base_model_file, material=sapien.render.RenderMaterial(base_color=[0.2, 0.4, 1.0, 1.0]))
        builder.add_convex_collision_from_file(balance_base_model_file)
        builder.initial_pose = sapien.Pose(p=[0.1, 0, 0])
        # self.balance_base = builder.build_kinematic(name="balance_base")
        self.balance_base = builder.build(name="balance_base")

        # Create balance beam
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[self.beam_size[0]/2, self.beam_size[1]/2, self.beam_size[2]/2])
        builder.add_box_visual(
            half_size=[self.beam_size[0]/2, self.beam_size[1]/2, self.beam_size[2]/2],
            material=sapien.render.RenderMaterial(base_color=[0.2, 0.4, 1.0, 1.0])
        )
        builder.initial_pose = sapien.Pose(p=[0.075, 0, 0.11])
        self.balance_beam = builder.build(name="balance_beam")

        # Create cubes with specific sizes
        self.cubes = []
        cube_sizes = [0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06]  # 4x0.03, 2x0.04, 2x0.05, 2x0.06
        for i, size in enumerate(cube_sizes):
            builder = self.scene.create_actor_builder()
            half_size = size / 2
            builder.add_box_collision(half_size=[half_size, half_size, half_size])
            
            # Map size [0.03, 0.06] to colors:
            # Small cubes (0.03) -> Pink [1.0, 0.4, 0.7, 1.0]
            # Large cubes (0.06) -> Red [1.0, 0.0, 0.0, 1.0]
            red = 1.0  # Keep red at maximum
            green = 0.4 - (size - 0.03) * (0.4 / 0.03)  # Decrease green for larger cubes
            blue = 0.7 - (size - 0.03) * (0.7 / 0.03)   # Decrease blue for larger cubes
            
            builder.add_box_visual(
                half_size=[half_size, half_size, half_size],
                material=sapien.render.RenderMaterial(base_color=[red, green, blue, 1.0])
            )
            builder.initial_pose = sapien.Pose(p=[0.3, 0.3, 0.5])
            cube = builder.build(name=f"cube_{i}")
            self.cubes.append(cube)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)

            # Initialize balance beam
            beam_pose = Pose.create_from_pq(
                p=torch.tensor([0.075, 0, 0.11], device=self.device).expand(b, 3),
                q=torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)
            )
            self.balance_beam.set_pose(beam_pose)
            self.balance_beam.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.balance_beam.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Initialize cubes with random positions
            for cube in self.cubes:
                pos = torch.zeros((b, 3), device=self.device)
                pos[..., 0] = torch.rand((b,), device=self.device) * 0.6 - 0.4  # x: [-0.4, 0.2]
                pos[..., 1] = torch.rand((b,), device=self.device) * 0.5 + 0.3  # y: [0.3, 0.8]
                pos[..., 2] = torch.rand((b,), device=self.device) * 0.2 + 0.02 # z: [0.02, 0.22]

                cube_pose = Pose.create_from_pq(
                    p=pos,
                    q=torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)
                )
                cube.set_pose(cube_pose)
                cube.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                cube.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def _quaternion_to_euler(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw] in radians.
        
        Args:
            q: Quaternion tensor [..., 4] in [w, x, y, z] format
            
        Returns:
            torch.Tensor: Euler angles [..., 3] in [roll, pitch, yaw] format
        """
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * math.pi / 2,  # use 90 degrees if out of range
            torch.asin(sinp)
        )
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=-1)

    def _check_beam_fallen(self, beam_pose: Pose) -> torch.Tensor:
        """Check if the beam has fallen based on multiple criteria.
        
        The beam is considered fallen if ANY of these conditions are met:
        1. z position is lower than 0.11
        2. abs of pitch angle or roll angle is larger than 30 degrees
        3. x,y position deviates from [0.075, 0] for more than 0.02
        
        Args:
            beam_pose: The pose of the balance beam
            
        Returns:
            torch.Tensor: Boolean tensor indicating if the beam has fallen (True) or not (False)
        """
        with torch.device(self.device):
            # Get position and orientation
            pos = beam_pose.p
            q = beam_pose.q
            
            # 1. Check z position
            z_too_low = pos[..., 2] < 0.10
            
            # 2. Check pitch and roll angles using direct quaternion to euler conversion
            euler_angles = self._quaternion_to_euler(q)
            pitch = euler_angles[..., 1]  # pitch is around y-axis
            roll = euler_angles[..., 0]   # roll is around x-axis
            angle_too_large = (torch.abs(pitch) > math.pi/6) | (torch.abs(roll) > math.pi/6)  # 30 degrees = pi/6
            
            # 3. Check x,y position deviation
            x_deviation = torch.abs(pos[..., 0] - 0.075)
            y_deviation = torch.abs(pos[..., 1])
            position_deviation = (x_deviation > 0.02) | (y_deviation > 0.02)
            
            # Beam is fallen if ANY of the conditions are met
            return z_too_low | angle_too_large | position_deviation

    def _check_cube_on_beam(self, cube_pos: torch.Tensor, cube_vel: torch.Tensor) -> torch.Tensor:
        """Check if a cube is properly placed on the beam.
        
        Criteria:
        1. z position should be higher than 0.12
        2. cube should be idle (no large velocity)
        3. x,y should be within beam bounds but not in the middle area
        
        Args:
            cube_pos: Position tensor of the cube [..., 3]
            cube_vel: Velocity tensor of the cube [..., 3]
            
        Returns:
            torch.Tensor: Boolean tensor indicating if the cube is on the beam (True) or not (False)
        """
        with torch.device(self.device):
            # 1. Check z position
            z_ok = cube_pos[..., 2] > 0.12
            
            # 2. Check if cube is idle (velocity magnitude should be small)
            velocity_magnitude = torch.norm(cube_vel, dim=-1)
            is_idle = velocity_magnitude < 0.01  # threshold for considering cube as idle
            
            # 3. Check x,y position
            # Define beam bounds
            x_center, y_center = 0.11, 0.0
            x_half_length, y_half_length = 0.025, 0.2
            
            # Define middle area bounds (to be excluded)
            middle_x_half = 0.02
            middle_y_half = 0.03
            
            # Check if within beam bounds
            within_beam = (
                (torch.abs(cube_pos[..., 0] - x_center) < x_half_length) &
                (torch.abs(cube_pos[..., 1] - y_center) < y_half_length)
            )
            
            # Check if in middle area (to be excluded)
            in_middle = (
                (torch.abs(cube_pos[..., 0] - x_center) < middle_x_half) &
                (torch.abs(cube_pos[..., 1] - y_center) < middle_y_half)
            )
            
            # Cube is on beam if it's within bounds but not in middle
            position_ok = within_beam & ~in_middle
            
            return z_ok & is_idle & position_ok

    def evaluate(self):
        """Evaluate the success of the task.
        
        Returns:
            Dict containing:
                - success: True when all cubes are on the beam and beam hasn't fallen
                - fail: True when the beam has fallen
                - cubes_on_beam: True for each cube that is properly placed
                - beam_fallen: True if beam has fallen
        """
        with torch.device(self.device):
            # Check if beam has fallen
            beam_fallen = self._check_beam_fallen(self.balance_beam.pose)

            # Check if cubes are on the beam
            cubes_on_beam = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for cube in self.cubes:
                cube_pos = cube.pose.p
                cube_vel = cube.get_linear_velocity()
                on_beam = self._check_cube_on_beam(cube_pos, cube_vel)
                cubes_on_beam = cubes_on_beam & on_beam

            return {
                "success": cubes_on_beam & ~beam_fallen,
                "fail": beam_fallen,
                "cubes_on_beam": cubes_on_beam,
                "beam_fallen": beam_fallen
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            beam_pose=self.balance_beam.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            # Add ground truth information if using state observations
            for i, cube in enumerate(self.cubes):
                obs[f"cube_{i}_pose"] = cube.pose.raw_pose
                obs[f"cube_{i}_vel"] = cube.get_linear_velocity()

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

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
            pose=look_at(eye=[0.7, 0.0, 0.7], target=[0, 0.2, 0.2]),
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        ) 