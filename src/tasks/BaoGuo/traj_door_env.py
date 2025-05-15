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


HANDLE_SIZE = [0.2, 0.02, 0.1]  # Size of the door
FRAME_SIZE = [0.5, 0.4, 0.01]  # Size of the door frame

@register_env("TrajDoor-v1", max_episode_steps=100)
class TrajDoorEnv(BaseEnv):
    """TrajDoor
        Task: try to open a door-like object on the table
        Goal: focus on the trajectory of the arm with related to the object
        
        Robot: Panda
        Object: Retangle-liked object
        Scene: Tabletop
        Action: joint position / trajectory
    """
    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        # Ensure the scene is passed correctly to TableSceneBuilder
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # Create an articulation between the door and the door frame
        articulation_builder = self.scene.create_articulation_builder()
        frame_link = articulation_builder.create_link_builder()
        frame_link.set_name("door_frame")
        frame_link.add_box_collision(half_size=FRAME_SIZE)
        frame_link.add_box_visual(
            half_size=FRAME_SIZE, 
            material=sapien.render.RenderMaterial(
                base_color=[0.4, 0.6, 0.8, 1],
            ),
        )
        frame_link.set_physx_body_type("kinematic")

        handle_link = articulation_builder.create_link_builder(parent=frame_link)
        handle_link.set_name("door")
        handle_link.add_box_collision(half_size=HANDLE_SIZE)
        handle_link.add_box_visual(
            half_size=HANDLE_SIZE,
            material=sapien.render.RenderMaterial(
                base_color=[1.0, 0.2, 0.4, 1],
            ),
        )
        handle_link.set_joint_properties(
            "revolute",
            limits=[[-np.pi / 2, np.pi / 2]],
            pose_in_parent=sapien.Pose(
                p=[0,0, FRAME_SIZE[2]], 
                q=R.from_euler('xyz', [np.deg2rad(90), np.deg2rad(90), np.deg2rad(90)]).as_quat()
                ),
            pose_in_child=sapien.Pose(
                p=[HANDLE_SIZE[0],0, -HANDLE_SIZE[2]], 
                q=[0, 0, 0, 1]
            ),
            friction=0.1,
            damping=0.5,
        )
        handle_link.set_physx_body_type("kinematic")

        articulation = articulation_builder.build(name="door_articulation")
        self.frame = articulation.get_links()[0]  # The first link is the frame
        self.door = articulation.get_links()[1]  # The second link is the door


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        with torch.device(self.device):
            b = len(env_idx)
            frame_init_p = torch.zeros((b, 3))
            frame_init_p[:, 0] = torch.rand((b,)) * 0.4 - 0.2
            frame_init_p[:, 1] = torch.rand((b,)) * 0.4 - 0.2
            frame_init_p[:, 2] = FRAME_SIZE[2]
            frame_pose = Pose.create_from_pq(p=frame_init_p, q=[0, 0, 0, 1])
            self.frame.set_pose(frame_pose)
            
            handle_init_p = torch.zeros((b, 3))
            handle_init_p[:, 0] = frame_init_p[:, 0] 
            handle_init_p[:, 1] = frame_init_p[:, 1]
            handle_init_p[:, 2] = frame_init_p[:, 2] + HANDLE_SIZE[0]
            handle_pose = Pose.create_from_pq(p=handle_init_p, q=[0, 0, 0, 1])
            self.door.set_pose(handle_pose)
            
            # self.frame.set_mass(100.0)
            # self.door.set_mass(1.0)

            # Define the goal position for the door (fully open)
            self.goal_door_angle = torch.tensor([np.pi / 2], device=self.device)  # 90 degrees open
        

    def evaluate(self):
        """Determine success/failure of the task
            The task is considered successful if the handler is opened to a certain angle.
            A success is defined as the angle of handler is within a certain range around the goal angle.
        """
        with torch.device(self.device):
            # Get the current door angle from the joint
            handle_joint = self.door.get_articulation()
            current_handle_angle = torch.tensor(handle_joint.get_qpos(), device=self.device)

            # Calculate the distance to the goal angle
            angle_to_goal = torch.abs(current_handle_angle - self.goal_door_angle)

            # Define a success threshold
            success_threshold = 0.1

            # Check if the door angle is within the success threshold
            success = (angle_to_goal < success_threshold).view(self.num_envs)  # Flatten to 1D

            # Return evaluation results
            return {"success": success, "angle_to_goal": angle_to_goal.view(self.num_envs)}
        

    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_door_angle,
        )
        
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning
            The reward is based on the angle difference to the goal and a success bonus.
            The reward is inversely proportional to the angle difference, with a large bonus for success.
        """
        with torch.device(self.device):
            # Initialize reward as a 1D tensor with shape (self.num_envs,)
            reward = torch.zeros(self.num_envs, device=self.device)

            # Calculate the angle difference to the goal
            angle_to_goal = info["angle_to_goal"].view(self.num_envs)

            # Reward is inversely proportional to the angle difference
            reward += 1.0 / (1.0 + angle_to_goal)

            # Add a large bonus for success
            success = info["success"].view(self.num_envs)
            reward = torch.where(success, reward + 10.0, reward)

            return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Normalize the dense reward"""
        max_reward = (
            13.0  # Maximum possible reward (success + all intermediate rewards)
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