import sapien
import torch
import numpy as np
from typing import Dict, Any, Union, List
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from scipy.spatial.transform import Rotation as R
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig

# Constants for object sizes and colors
CONTAINER_SIZE = [0.03, 0.03, 0.05, 0.005]  # Length, width, height, thickness
TARGET_CUBOID_SIZE = [0.028, 0.048, 0.028]  # Length, width, height
CUBOID_SIZE = [[0.05, 0.05, 0.06],
               [0.023, 0.023, 0.023],
               [0.04, 0.03, 0.05]]  # Length, width, height
COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB colors


@register_env("InsertCuboid-v1", max_episode_steps=150)
class InsertCuboidEnv(BaseEnv):
    """
    Task: Select the most suitable cuboid for the given container based on its size. 
    Precisely align and insert the cuboid into the container to ensure a perfect fit.

    Difficulty can be adjusted by changing the number of cuboids,
    changing the cuboid size(making the cuboids cubes may make the task easier), 
    adjusting the cuboid initial orientation(whether the robot need to rotate the cuboid before inserting it), 
    or adding other constraints(e.g. color, texture).
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", num_envs=1, reconfiguration_freq=None, **kwargs):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)
        self.selected_cuboid = None
        print("Initialized selected_cuboid as None")  # 调试信息

    def _load_scene(self, options: dict):
        """Load the scene with objects and containers."""
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.target_cuboid = []
        self.cuboid = []
        self.container = []

        # Create container
        self._create_container(index=0, color=[0.5, 0.5, 0.5])  # Gray container

        # Create target cuboid
        self._create_cuboid(
            size=TARGET_CUBOID_SIZE,
            color=[0, 1, 0],
            name="target_cuboid",
            position=[0.1, 0, 0.03],
            collection=self.target_cuboid,
        )

        # Create distractor cuboids
        distractor_positions = [[-0.1, 0, 0.03], [0, 0.1, 0.03], [0, -0.1, 0.03]]
        for i, size in enumerate(CUBOID_SIZE):
            self._create_cuboid(
                size=size,
                color=[0, 1, 0],
                name=f"distractor_cuboid_{i}",
                position=distractor_positions[i],
                collection=self.cuboid,
            )

    def _create_cuboid(self, size, color, name, position, collection):
        """Helper function to create a cuboid."""
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[s / 2 for s in size], density=500)
        builder.add_box_visual(
            half_size=[s / 2 for s in size],
            material=sapien.render.RenderMaterial(base_color=color + [1.0]),
        )
        obj = builder.build(name=name)
        obj.set_pose(sapien.Pose(p=position))
        collection.append(obj)

    def _create_container(self, index, color):
        """Helper function to create a container."""
        builder = self.scene.create_actor_builder()

        # Walls
        wall_thickness = CONTAINER_SIZE[3] / 4
        wall_height = CONTAINER_SIZE[2]
        wall_half_height = wall_height / 2

        # Front wall
        builder.add_box_collision(
            half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
            pose=sapien.Pose(p=[0, CONTAINER_SIZE[1] - wall_thickness, wall_half_height]),
        )
        builder.add_box_visual(
            half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
            pose=sapien.Pose(p=[0, CONTAINER_SIZE[1] - wall_thickness, wall_half_height]),
            material=sapien.render.RenderMaterial(base_color=color + [1.0]),
        )

        # Back wall
        builder.add_box_collision(
            half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
            pose=sapien.Pose(p=[0, -CONTAINER_SIZE[1] + wall_thickness, wall_half_height]),
        )
        builder.add_box_visual(
            half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
            pose=sapien.Pose(p=[0, -CONTAINER_SIZE[1] + wall_thickness, wall_half_height]),
            material=sapien.render.RenderMaterial(base_color=color + [1.0]),
        )

        # Left wall
        builder.add_box_collision(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[-CONTAINER_SIZE[0] + wall_thickness, 0, wall_half_height]),
        )
        builder.add_box_visual(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[-CONTAINER_SIZE[0] + wall_thickness, 0, wall_half_height]),
            material=sapien.render.RenderMaterial(base_color=color + [1.0]),
        )
         # Right wall
        builder.add_box_collision(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[CONTAINER_SIZE[0] - wall_thickness, 0, wall_half_height]),
        )
        builder.add_box_visual(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[CONTAINER_SIZE[0] - wall_thickness, 0, wall_half_height]),
            material=sapien.render.RenderMaterial(base_color=color + [1.0]),
        )

        # Build container
        container = builder.build_static(name=f"container_{index}")
        self.container.append(container)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Randomize object positions at the start of each episode."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Set container position at the center
            container_pos = torch.tensor([0.0, 0.0, 0.005], device=self.device)  # Center position
            self.container[0].set_pose(Pose.create_from_pq(p=container_pos))

            # Define workspace boundaries for random placement
            workspace_radius = 0.3  # Maximum reach of the robot arm
            min_distance = 0.05  # Minimum distance between objects to avoid overlap

            # Randomize positions for the target cuboid and distractor cuboids
            all_positions = []
            for obj in self.target_cuboid + self.cuboid:
                valid_position = False
                while not valid_position:
                    # Generate random position within the workspace
                    x = (torch.rand(1, device=self.device) - 0.5) * 2 * workspace_radius
                    y = (torch.rand(1, device=self.device) - 0.5) * 2 * workspace_radius
                    z = torch.tensor([0.03], device=self.device)  # Slightly above the table

                    # Check for minimum distance from other objects
                    new_pos = torch.tensor([x.item(), y.item(), z.item()], device=self.device)
                    valid_position = all(
                        torch.linalg.norm(new_pos[:2] - pos[:2]) > min_distance for pos in all_positions
                    )

                # Save the valid position and set the object's pose
                all_positions.append(new_pos)
                obj.set_pose(Pose.create_from_pq(p=new_pos))

            # Randomize orientation for all cuboids
            for obj in self.target_cuboid + self.cuboid:
                random_yaw = torch.rand(1, device=self.device) * 2 * np.pi
                quaternion = torch.tensor(
                    R.from_euler('z', random_yaw.cpu().numpy(), degrees=False).as_quat(),
                    device=self.device
                )
                obj.set_pose(Pose.create_from_pq(
                    p=obj.pose.p,
                    q=quaternion
                ))

            # Set linear and angular velocities to zero for dynamic objects only
            for obj in self.target_cuboid + self.cuboid:
                obj.set_linear_velocity(torch.zeros((3,), device=self.device))
                obj.set_angular_velocity(torch.zeros((3,), device=self.device))


    def evaluate(self):
        """Determine success/failure of the task."""
        print("Entering evaluate method...")  # Debug information
        with torch.device(self.device):
            # Get the positions of the target cuboid and the container
            target_pos = self.target_cuboid[0].pose.p.cpu().numpy().squeeze()  # Convert to NumPy array and remove extra dimensions
            container_pos = self.container[0].pose.p.cpu().numpy().squeeze()  # Convert to NumPy array and remove extra dimensions

            # Debug information: print positions and shapes
            # print(f"target_pos: {target_pos}, shape: {target_pos.shape}")
            # print(f"container_pos: {container_pos}, shape: {container_pos.shape}")

            # Check if the positions are 3D
            if target_pos.shape[0] != 3 or container_pos.shape[0] != 3:
                raise ValueError(f"Invalid position shape: target_pos={target_pos}, container_pos={container_pos}")

            # Check if the target cuboid is perfectly inserted
            xy_aligned = (
                np.all(np.abs(target_pos[:2] - container_pos[:2]) < 0.002)  # Check alignment in x and y
            )
            z_aligned = target_pos[2] <= container_pos[2]  # Check alignment in z

            # Debug information: print alignment status
            # print(f"xy_aligned: {xy_aligned}, z_aligned: {z_aligned}")

            # Determine if the task is successful
            success = xy_aligned and z_aligned

            # Check if any distractor cuboid is in the container
            distractor_in_container = False
            for cuboid in self.cuboid:
                cuboid_pos = cuboid.pose.p.cpu().numpy().squeeze()  # Convert to NumPy array and remove extra dimensions
                cuboid_half_size = np.array([size / 2 for size in CUBOID_SIZE[self.cuboid.index(cuboid)]])

                # Container boundaries
                container_min = container_pos - np.array([CONTAINER_SIZE[0] / 2, CONTAINER_SIZE[1] / 2, 0])
                container_max = container_pos + np.array([CONTAINER_SIZE[0] / 2, CONTAINER_SIZE[1] / 2, CONTAINER_SIZE[2]])

                # Distractor cuboid boundaries
                cuboid_min = cuboid_pos - cuboid_half_size
                cuboid_max = cuboid_pos + cuboid_half_size

                # Check if the cuboid is inside the container
                overlap_x = cuboid_min[0] < container_max[0] and cuboid_max[0] > container_min[0]
                overlap_y = cuboid_min[1] < container_max[1] and cuboid_max[1] > container_min[1]
                overlap_z = cuboid_min[2] < container_max[2] and cuboid_max[2] > container_min[2]

                if overlap_x and overlap_y and overlap_z:
                    distractor_in_container = True
                    break

            # Debug information: print distractor status
            # print(f"distractor_in_container: {distractor_in_container}")

            # Convert success and distractor_in_container to 1D PyTorch tensors
            success_tensor = torch.tensor([success], device=self.device, dtype=torch.bool)
            distractor_tensor = torch.tensor([distractor_in_container], device=self.device, dtype=torch.bool)

            return {
                "success": success_tensor,
                "distractor_in_container": distractor_tensor,
            }
        
    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task."""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            # Add positions of all cuboids and the container
            for i, cuboid in enumerate(self.target_cuboid + self.cuboid):
                obs[f"cuboid_{i}_pose"] = cuboid.pose.raw_pose

            obs["container_pose"] = self.container[0].pose.raw_pose

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning."""
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)

            # Reward for perfect insertion
            if info["success"]:
                reward += 5.0  # Perfect insertion

            # Penalty for distractor cuboid in the container
            if info["distractor_in_container"]:
                reward -= 3.0  # Distractor penalty

            return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward"""
        max_reward = 5.0 
        min_reward = - 3.0

        raw_reward = self.compute_dense_reward(obs, action, info)

        normalized_reward = (raw_reward - min_reward) / (max_reward - min_reward)
        return normalized_reward
    

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=256,
            height=256,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )

        # Side view camera
        side_camera = CameraConfig(
            "side_camera",
            pose=look_at(eye=[1.0, -1.0, 1.0], target=[0, 0, 0.2]),
            width=256,
            height=256,
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
            pose=look_at(eye=[1.0, -1.2, 1.2], target=[0, 0, 0.1]),
            width=1024,
            height=1024,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
