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
CONTAINER_SIZE = [0.1, 0.1, 0.05, 0.005]  # Length, width, height, thickness
CAPSULE_SIZE = [0.025, 0.05]
SPHERE_SIZE = [0.03, 0.03, 0.03]
COLORS = [
    [1.0, 0.2, 0.2, 1.0],  # Red
    [0.2, 0.6, 1.0, 1.0],  # Blue
    [0.2, 0.8, 0.2, 1.0],  # Green
]

# Object positions
BANANA_POSITIONS = [[-0.2, 0, 1.05], [0.2, 0, 1.05]]  # Two bananas

@register_env("FoodSorting-v1", max_episode_steps=150)
class FoodSortingEnv(BaseEnv):
    """
    Task instruction: Pick all the edible items (bananas) and place them in the red container,
    and all the inedible items (capsule/sphere) in the blue container.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", num_envs=1, reconfiguration_freq=None, **kwargs):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)

    def _load_scene(self, options: dict):
        """Load the scene with objects and containers."""
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.food = []
        self.others = []
        self.container = []

        # Create bananas
        for i, pos in enumerate(BANANA_POSITIONS):  # Two bananas
            self._create_object(
                collision_file="/banana/collision.obj",
                visual_file="/banana/visual.glb",
                name=f"banana_{i}",
                position=pos,
                collection=self.food,
            )

        # Create capsule
        self._create_capsule_or_sphere(
            name="capsule",
            size=CAPSULE_SIZE,  # [0.01, 0.02]
            color=COLORS[0], 
            collection=self.others,
        )

        # Create sphere
        self._create_capsule_or_sphere(
            name="sphere",
            size=SPHERE_SIZE,  # [0.01, 0.01, 0.01]
            color=COLORS[1], 
            collection=self.others,
        )

        # Create containers
        for i, color in enumerate(COLORS):
            self._create_container(index=i, color=color)

    def _create_capsule_or_sphere(self, name, size, color, collection):
        """Helper function to create a capsule or sphere."""
        builder = self.scene.create_actor_builder()
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0]))

        if name == "capsule":
            builder.add_capsule_collision(radius=size[0], half_length=size[1], density=500)
            builder.add_capsule_visual(radius=size[0], half_length=size[1], material=sapien.render.RenderMaterial(base_color=color))
        elif name == "sphere":
            builder.add_sphere_collision(radius=size[0], density=500)
            builder.add_sphere_visual(radius=size[0], material=sapien.render.RenderMaterial(base_color=color))
        else:
            raise ValueError(f"Unsupported object type: {name}")

        obj = builder.build(name=name)
        collection.append(obj)

    def _create_object(self, collision_file, visual_file, name, position, collection):
        """Helper function to create an object."""
        builder = self.scene.create_actor_builder()

        scale = np.array([1 / 1.2, 1 / 1.2, 1 / 1.2], dtype=np.float32)
        builder.add_convex_collision_from_file(filename=collision_file, scale=scale)
        builder.add_visual_from_file(filename=visual_file, scale=scale)

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
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Back wall
        builder.add_box_collision(
            half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
            pose=sapien.Pose(p=[0, -CONTAINER_SIZE[1] + wall_thickness, wall_half_height]),
        )
        builder.add_box_visual(
            half_size=[CONTAINER_SIZE[0], wall_thickness, wall_half_height],
            pose=sapien.Pose(p=[0, -CONTAINER_SIZE[1] + wall_thickness, wall_half_height]),
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Left wall
        builder.add_box_collision(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[-CONTAINER_SIZE[0] + wall_thickness, 0, wall_half_height]),
        )
        builder.add_box_visual(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[-CONTAINER_SIZE[0] + wall_thickness, 0, wall_half_height]),
            material=sapien.render.RenderMaterial(base_color=color),
        )
         # Right wall
        builder.add_box_collision(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[CONTAINER_SIZE[0] - wall_thickness, 0, wall_half_height]),
        )
        builder.add_box_visual(
            half_size=[wall_thickness, CONTAINER_SIZE[1], wall_half_height],
            pose=sapien.Pose(p=[CONTAINER_SIZE[0] - wall_thickness, 0, wall_half_height]),
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Build container
        container = builder.build_static(name=f"container_{index}")
        self.container.append(container)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Randomize object positions at the start of each episode."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Move containers to their designated positions
            for i, container in enumerate(self.container):
                container_pos = torch.tensor([0.3 * (i - 1), 0.0, 0.005], device=self.device)  # Example positions
                container.set_pose(Pose.create_from_pq(p=container_pos))

            # Container dimensions and safe distance
            container_width, container_depth, container_height, wall_thickness = CONTAINER_SIZE
            container_radius = max(container_width, container_depth) / 2 + wall_thickness
            item_radius = max(CAPSULE_SIZE[0], SPHERE_SIZE[0])  # Assuming items are capsules or spheres
            min_distance = 2 * item_radius + 0.02
            max_distance = 0.4 - container_radius - 2 * item_radius - 0.02

            # Define angles for item placement (randomized for each object)
            num_items = len(self.food + self.others)
            angles = torch.rand((b, num_items), device=self.device) * 2 * np.pi  # Random angles for all items

            # Randomize distances for items
            distances = min_distance + (max_distance - min_distance) * torch.rand((b, num_items), device=self.device)

            # Place items
            for i, obj in enumerate(self.food + self.others):
                item_pos = torch.zeros((b, 3), device=self.device)
                item_pos[..., 0] = distances[:, i] * torch.cos(angles[:, i])
                item_pos[..., 1] = distances[:, i] * torch.sin(angles[:, i])
                item_pos[..., 2] = 0.02 if i < len(self.food) else CAPSULE_SIZE[0]  # Slightly above the table

                # Introduce noise on location
                item_pos[..., :2] += (torch.rand((b, 2), device=self.device)) * 0.04 - 0.02

                # Introduce noise on rotation
                random_yaw = torch.rand((b,), device=self.device) * 2 * np.pi
                random_pitch = (torch.rand((b,), device=self.device)) * 0.2 - 0.1

                # Convert random_pitch and random_yaw to NumPy arrays
                random_pitch_np = random_pitch.cpu().numpy()
                random_yaw_np = random_yaw.cpu().numpy()

                # Convert Euler angles to quaternion
                quaternion = torch.tensor(
                    R.from_euler('xyz', np.column_stack((random_pitch_np, np.zeros(b), random_yaw_np)), degrees=False).as_quat(),
                    device=self.device
                )

                obj.set_pose(Pose.create_from_pq(
                    p=item_pos,
                    q=quaternion
                ))

                # Set linear and angular velocities
                obj.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                obj.set_angular_velocity(torch.zeros((b, 3), device=self.device))

        
    def evaluate(self):
        """
        Determine success/failure of the task.

        This function evaluates whether the task of sorting items into their respective containers 
        has been successfully completed. The task involves placing food items (e.g., bananas) into 
        the red container and non-food items (e.g., capsules and spheres) into the green container. 
        The function performs the following steps:

        1. **Retrieve Positions**:
           - Collect the positions of all items (`self.food` and `self.others`) and containers (`self.container`).
           - Positions are represented as 3D coordinates (x, y, z).

        2. **Define Container Boundaries**:
           - Define the height range of the containers using `container_bottom_height` and `container_top_height`.

        3. **Initialize Tracking Arrays**:
           - Create boolean arrays to track whether each item is in the correct container (`items_in_correct_container`) 
             or in the wrong container (`items_in_wrong_container`).

        4. **Check Item Placement**:
           - For each item and each container:
             - Compute the 2D distance (`dist_xy`) between the item's position and the container's position in the XY plane.
             - Check if the item is within the container's XY boundaries (`inside_container`).
             - Check if the item's height is within the container's height range (`proper_height`).
             - Combine these checks to determine if the item is inside the container (`in_this_container`).
             - Determine if the item is in the correct container:
               - Food items (e.g., bananas) should be in the red container (index 0).
               - Non-food items (e.g., capsules and spheres) should be in the green container (index 1).
             - Update the tracking arrays based on whether the item is in the correct or wrong container.

        5. **Count Items**:
           - Count the number of items in the correct containers (`correct_count`) and in the wrong containers (`wrong_count`).

        6. **Determine Success**:
           - The task is considered successful if all items are in their correct containers (`success`).

        7. **Return Results**:
           - Return a dictionary containing:
             - `success`: A boolean indicating whether the task was successful.
             - `items_correct`: A boolean array indicating which items are in the correct containers.
             - `items_wrong`: A boolean array indicating which items are in the wrong containers.
             - `correct_count`: The number of items in the correct containers.
             - `wrong_count`: The number of items in the wrong containers.
        """
        with torch.device(self.device):
            # Get the positions of items and containers
            item_positions = torch.stack([item.pose.p for item in self.food + self.others])  # [num_items, b, 3]
            container_positions = torch.stack([container.pose.p for container in self.container])  # [3, b, 3]

            # Container height
            container_bottom_height = 0.005
            container_top_height = container_bottom_height + CONTAINER_SIZE[2]

            # Initialize arrays to track items in correct and wrong containers
            items_in_correct_container = torch.zeros((len(self.food + self.others), self.num_envs), dtype=torch.bool, device=self.device)
            items_in_wrong_container = torch.zeros((len(self.food + self.others), self.num_envs), dtype=torch.bool, device=self.device)

            for i, item in enumerate(self.food + self.others):
                for j, container in enumerate(self.container):
                    # Check if the item is inside the container's XY plane
                    dist_xy = torch.linalg.norm(item_positions[i, :, :2] - container_positions[j, :, :2], dim=1)
                    inside_container = (dist_xy < (CONTAINER_SIZE[0] - 0.01)).bool()  # Ensure bool type

                    # Check if the item is within the height of the container
                    proper_height = ((item_positions[i, :, 2] > container_bottom_height) & 
                                    (item_positions[i, :, 2] < container_top_height)).bool()  # Ensure bool type

                    # Check if the item is inside the container
                    in_this_container = (inside_container & proper_height).bool()  # Ensure bool type

                    # Determine if the item is in the correct container
                    if i < len(self.food):  # Food items (bananas)
                        correct_container = (j == 0)  # Red container
                    else:  # Other items (capsule/sphere)
                        correct_container = (j == 1)  # Green container

                    correct_container = torch.tensor(correct_container, dtype=torch.bool, device=self.device)  # Ensure bool type

                    items_in_correct_container[i] |= in_this_container & correct_container
                    items_in_wrong_container[i] |= in_this_container & ~correct_container

            # Count the number of items in correct and wrong containers
            correct_count = torch.sum(items_in_correct_container, dim=0)
            wrong_count = torch.sum(items_in_wrong_container, dim=0)

            # Determine success
            success = correct_count == len(self.food + self.others)

            return {
                "success": success,
                "items_correct": items_in_correct_container,
                "items_wrong": items_in_wrong_container,
                "correct_count": correct_count,
                "wrong_count": wrong_count,
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            # Add items' and containers' positions
            for i, item in enumerate(self.food + self.others):
                obs[f"item_{i}_pose"] = item.pose.raw_pose
                obs[f"item_{i}_vel"] = item.linear_velocity

            for i, container in enumerate(self.container):
                obs[f"container_{i}_pose"] = container.pose.raw_pose

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
            """Compute a dense reward signal to guide learning"""
            with torch.device(self.device):
                reward = torch.zeros(self.num_envs, device=self.device)

                # For each item in the correct container, reward +2
                items_correct = info["items_correct"]
                for i in range(len(self.food + self.others)):
                    reward = torch.where(items_correct[i], reward + 2.0, reward)

                # For each item in the wrong container, reward -1
                items_wrong = info["items_wrong"]
                for i in range(len(self.food + self.others)):
                    reward = torch.where(items_wrong[i], reward - 1.0, reward)

                # For each item below the table, reward -2
                item_positions = torch.stack([item.pose.p for item in self.food + self.others])
                table_height = 0.0  # Assuming the table is at height 0.0
                for i in range(len(self.food + self.others)):
                    item_below_table = item_positions[i, :, 2] < table_height - 0.05
                    reward = torch.where(item_below_table, reward - 2.0, reward)

                # For final success, reward +5
                success = info["success"]
                reward = torch.where(success, reward + 5.0, reward)

                return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward"""
        max_reward = 5.0 + (2.0 * len(self.food + self.others)) 
        min_reward = - (2.0 * len(self.food + self.others))

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
