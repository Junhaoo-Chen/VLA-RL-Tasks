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


@register_env("CylinderPushUp-v1", max_episode_steps=200)
class CylinderPushUpEnv(BaseEnv):
    """
    **Task Description:**
    Push the cylinder onto the blue cube through the ramp, and make sure the cylinder stays still on top of the blue cube.

    **Instruction:**
    "Push the cylinder onto the blue cube and make it stay still without grasping it."

    **Randomization:**
    - the xy position and z-axis rotation of the cylinder are randomized

    **Success Condition:**
    - The cylinder is pushed into the target region and stays still.
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent = Union[Panda, Fetch]

    # Parameters
    cylinder_radius = 0.02
    cylinder_half_length = 0.04

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        # Table
        self.table_scene = TableSceneBuilder(
            env=self,
        )
        self.table_scene.build()
        # Create cylinder
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
        )
        builder.add_cylinder_visual(
            radius=self.cylinder_radius,
            half_length=self.cylinder_half_length,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
        )
        builder.initial_pose = sapien.Pose()
        builder.initial_pose = sapien.Pose(p=[0.0, -0.1, 0.02], q=[1, 0, 0, 0])
        self.cylinder = builder.build_dynamic(name="cylinder")

        # Create ramp
        ramp_model_file = os.path.join(
            os.path.dirname(__file__), "assets", "ramp.obj"
        )
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename=ramp_model_file,
        )
        builder.add_visual_from_file(
            filename=ramp_model_file,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        builder.initial_pose = sapien.Pose(p=[-0.05, -0.3, 0], q=[0, 1, 0, 0])
        self.ramp = builder.build_kinematic(name="ramp")

        # Create the blue cube whose top surface is the target region
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[0.05] * 3,
        )
        builder.add_box_visual(
            half_size=[0.05] * 3,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0.35, 0.05], q=[1, 0, 0, 0])
        self.target_box = builder.build_kinematic(name="target_box")
        
        # Test object
        # builder = self.scene.create_actor_builder()
        # builder.add_cylinder_collision(
        #     radius=self.cylinder_radius,
        #     half_length=self.cylinder_half_length,
        # )
        # builder.add_cylinder_visual(
        #     radius=self.cylinder_radius,
        #     half_length=self.cylinder_half_length,
        #     material=sapien.render.RenderMaterial(base_color=[0.5, 0, 0.5, 1]),
        # )
        # builder.initial_pose = sapien.Pose()
        # builder.initial_pose = sapien.Pose(p=[0.0, 0.2, 0.2], q=[1, 0, 0, 0])
        # self.test_obj = builder.build_dynamic(name="test_object")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)
            # Randomize the cylinder pose
            p = torch.zeros((b, 3))
            p[:, 0] = torch.rand(b) * 0.2 - 0.1
            p[:, 1] = torch.rand(b) * 0.2 - 0.3
            p[:, 2] = self.cylinder_radius
            # Generate a random z-axis rotation
            angles = torch.rand(b) * 2 * math.pi  # Random angles in [0, 2π)
            q = torch.zeros((b, 4))
            q[:, 0] = torch.cos(angles / 2)
            q[:, 3] = torch.sin(angles / 2)
            obj_pose = Pose.create_from_pq(p=p, q=q)
            self.cylinder.set_pose(obj_pose)
            self.cylinder.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.cylinder.set_angular_velocity(torch.zeros((b, 3), device=self.device))

    def evaluate(self):
        """Evaluate the success of the task and two subgoals: getting into the target region and stopping.

        Criteria:
        1. Inside target region:
            - The xy position cylinder is inside the target region, which is the top surface of the blue cube.
            - The z position of the cylinder is higher than the cube.
        2. Stopped:
            - The speed of the cylinder is less than 0.01 m/s.
        3. Success:
            - The cylinder is inside the target region and stopped.

        Returns:
            - success: A boolean indicating whether the task is successful.
            - is_cylinder_in_target: A boolean indicating whether the cylinder is inside the target region.
            - is_cylinder_still: A boolean indicating whether the cylinder is stopped.
        """
        # success is achieved when the cylinder's position is within the target region while it is not moving
        is_cylinder_in_target = (
            self.cylinder.pose.p[..., 0] > -0.1 and
            self.cylinder.pose.p[..., 0] < 0.1 and
            self.cylinder.pose.p[..., 1] < 0.3 and
            self.cylinder.pose.p[..., 1] > 0.4 and
            self.cylinder.pose.p[..., 2] > 0.1
        )
        is_cylinder_still = torch.norm(self.cylinder.get_linear_velocity(), dim=-1) < 0.01
        success = is_cylinder_in_target and is_cylinder_still
        return {
            "success": success,
            "is_cylinder_in_target": is_cylinder_in_target,
            "is_cylinder_still": is_cylinder_still,
        }
    
    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            # Add ground truth information if using state observations
            obs[f"cylinder_pose"] = self.cylinder.pose.raw_pose
            # obs[f"target_x_region"] = torch.tensor(
            #     [0.2, 0.3], device=self.device
            # )
            # obs[f"target_y_region"] = torch.tensor(
            #     [-0.4, -0.3], device=self.device
            # )
            # obs[f"target_z_region"] = torch.tensor(
            #     [0.1, 0.2], device=self.device
            # )

        return obs

    def compute_dense_reward(self, obs, action, info):
        """
        Compute the dense reward for the task.

        Reward components:
        - Encourages the cylinder to move closer to the target region (in xy).
        - Rewards lifting the cylinder to the correct height.
        - Gives a bonus for being inside the target region.
        - Rewards the cylinder for being still inside the target region.
        - Penalizes any use of the gripper.
        """
        with torch.device(self.device):
            reward = torch.zeros(self.num_envs, device=self.device)
            success = info["success"]
            is_cylinder_in_target = info["is_cylinder_in_target"]
            is_cylinder_still = info["is_cylinder_still"]

            # Reward for moving the cylinder closer to the target region horizontally (xy position), max 1
            cylinder_pos = self.cylinder.pose.p
            target_xy = torch.tensor([0.25, -0.35], device=self.device).expand(self.num_envs, 2)
            distance_to_target = torch.linalg.norm(cylinder_pos[..., :2] - target_xy, dim=-1)
            reward += torch.exp(-1.0 * distance_to_target)

            # Reward for lifting up the cylinder to get closer to the target height, max 3
            height_reward = (cylinder_pos[..., 2] - self.cylinder_radius) * 30
            height_reward = torch.clamp(height_reward, max=3.0)
            reward += height_reward

            # Reward for getting into the target, 3
            reward = torch.where(is_cylinder_in_target, reward + 3.0, reward)

            # Reward for slowing down the cylinder inside the target region
            # Only apply to those in target region
            velocities = self.cylinder.get_linear_velocity()
            slow_bonus = 3 * torch.exp(-1.0 * torch.linalg.norm(velocities, dim=-1))
            reward = torch.where(is_cylinder_in_target, reward + slow_bonus, reward)

            # Punish if the gripper action is not 0
            gripper_action = action[..., -1]
            gripper_penalty = -torch.abs(gripper_action) * 10
            reward += gripper_penalty

            return reward
    
    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 10.0
    
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
            pose=look_at(eye=[0.7, 0.3, 0.6], target=[0, 0, 0.1]),
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
