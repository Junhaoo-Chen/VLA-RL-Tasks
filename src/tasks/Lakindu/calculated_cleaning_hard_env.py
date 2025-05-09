# trash_collection_multiple_shapes_env.py
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
from .custom_shapes import *

def check_whether_object_not_on_table(table_scene, obj_pos, tolerance: float = 0.02) -> bool:
    """
    Check if an object is not on the table.
    
    Args:
        obj: The object to check
        tolerance: Vertical distance tolerance to consider the object "on" the table
        table_scene: table_Scene object
        
    Returns:
        bool: True if the object is on the table, False otherwise
    """
    
    
    # Table dimensions are already in world coordinates
    table_pos = table_scene.table.pose.p

    table_length = table_scene.table_length
    # table_height = table_scene.table_height
    table_width = table_scene.table_width
    
    is_at_table_height = abs(obj_pos[:, :, 2] - 0) < tolerance
    
    # Check if object is within table boundaries
    # Use table's actual position from its pose
    is_within_x = abs(obj_pos[:, :, 0] - table_pos[:, 0]) <= table_length / 2
    is_within_y = abs(obj_pos[:, :, 1] - table_pos[:, 1]) <= table_width / 2

    return ~(is_at_table_height & is_within_x & is_within_y)


@register_env("CalculatedCleaningHard-v1", max_episode_steps=100)
class CalculatedCleaningHard(BaseEnv):
    """
    Task: Use nearby objects to throw all the unreachable objects out. Make sure that all the throwable objects aren't thrown off the table.
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        n_obj=4,
        n_throwable=4,
        **kwargs,
    ):
        # Set reconfiguration frequency - for single env, reconfigure every time
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
                
        self.n_obj = n_obj  
        self.n_throwable = n_throwable
        
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        # Create table scene
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        self.objects = create_random_trash(
            scene=self.scene,
            num_pieces=self.n_obj, 
            min_size=0.04,  
            max_size=0.08,
            name_suffix="fixed"
        )

        self.throwable_objects = create_random_trash(
            scene=self.scene,
            num_pieces=self.n_throwable, 
            min_size=0.04,  
            max_size=0.08,
            name_suffix="throwable"
        )
    
        

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Initialize table scene
        self.table_scene.initialize(env_idx)
        
        # Generate batch dimension based on env_idx
        with torch.device(self.device):
            b = len(env_idx)
            
            # spawn yeetable objects pieces in random locations
            for obj in self.objects:
                x = np.random.uniform(0.2, 0.4)    # might have to be adjustedd
                y = np.random.uniform(-0.6, 0.8)   
                z = 0.01                           # fixed height

                # Random rotation
                angle = np.random.uniform(0, 2 * np.pi)
                axis = np.random.uniform(-1, 1, 3)
                axis = axis / np.linalg.norm(axis)
                sin_a = np.sin(angle / 2)
                cos_a = np.cos(angle / 2)
                quat = [cos_a, axis[0] * sin_a, axis[1] * sin_a, axis[2] * sin_a]

                obj.set_pose(sapien.Pose([x, y, z], quat))
        

            # spawn throwable objects near robot
            for obj in self.throwable_objects:
                x = np.random.uniform(-0.4, 0)    # might have to be adjustedd
                y = np.random.uniform(-0.2, 0.2)   
                z = 0.01                           # fixed height

                # Random rotation
                angle = np.random.uniform(0, 2 * np.pi)
                axis = np.random.uniform(-1, 1, 3)
                axis = axis / np.linalg.norm(axis)
                sin_a = np.sin(angle / 2)
                cos_a = np.cos(angle / 2)
                quat = [cos_a, axis[0] * sin_a, axis[1] * sin_a, axis[2] * sin_a]

                obj.set_pose(sapien.Pose([x, y, z], quat))


    def evaluate(self):
        """Determine success/failure of the task"""
        

        with torch.device(self.device):
            obj_pos = torch.stack([obj.pose.p for obj in self.objects]) #[n_trash, n_batch, n_pos]
            throwable_obj_pos = torch.stack([obj.pose.p for obj in self.throwable_objects])
            
            whether_throwable_object_not_on_table = check_whether_object_not_on_table(table_scene=self.table_scene, obj_pos=throwable_obj_pos, tolerance=0.05)
            whether_object_not_on_table = check_whether_object_not_on_table(table_scene=self.table_scene, obj_pos=obj_pos, tolerance=0.05)
        
            success_1 = torch.all(whether_object_not_on_table, dim=0)
            success_2 = torch.all(~whether_throwable_object_not_on_table, dim=0)
            success = success_1 & success_2

            num_throwm = torch.sum(whether_object_not_on_table, dim=0)  # Shape: [n_batch]
            num_throwable_not_thrown = torch.sum(~whether_throwable_object_not_on_table, dim=0)

        return {"success": success,
                "num_throwm": num_throwm,
                "num_throwable_not_thrown": num_throwable_not_thrown,
                "obj_pos": obj_pos,
                }
        
    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose
        )
        if self.obs_mode_struct.use_state:
            for i, obj in enumerate(self.objects):
                obs[f"trash_{i}_pose"] = obj.pose.raw_pose
                obs[f"trash_{i}_vel"] = obj.linear_velocity
        return obs
        
        
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning"""
        success = info['success']
        num_throwm = info['num_throwm']
        num_throwable_not_thrown = info['num_throwable_not_thrown']
        obj_pos = info['obj_pos']

        table_centre = self.table_scene.table.pose.p 

        distances = torch.linalg.norm(obj_pos[:, :, :2] - table_centre[:, :2], dim=-1)
        
        # distance reward
        distances_exp = (1 - torch.exp(-1.5 * distances))
        distances_reward = torch.sum(distances_exp, dim=0)  
        
        max_dist_reward = self.n_obj # max is n_obj*k

        # progression reward
        progress_reward =  num_throwm * 1.5
        max_progress_reward = self.n_obj * 1.5 # max is n_obj*k

        # "throwable objects not thrown away from the table" reward
        pow_term = torch.pow(2, num_throwable_not_thrown)
        throwable_obj_not_thrown_reward = 2 * (pow_term - 1.0)
        max_throwable_obj_not_thrown_reward = self.n_throwable * (2 * (torch.pow(2, torch.tensor(self.n_throwable)) - 1.0))
        
        # rescale
        throwable_obj_not_thrown_reward = throwable_obj_not_thrown_reward/max_throwable_obj_not_thrown_reward * 3
        max_throwable_obj_not_thrown_reward = self.n_throwable * 3

        # maximum possible reward
        self.max_reward = (max_dist_reward + max_progress_reward + max_throwable_obj_not_thrown_reward)*2 

        # success reward
        total_reward = distances_reward + progress_reward + throwable_obj_not_thrown_reward
        total_reward = torch.where(success, self.max_reward*2, total_reward)
              
        return total_reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward"""
        return self.compute_dense_reward(obs, action, info) / self.max_reward

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        # Top-down camera view
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=128,
            height=128,
            fov=np.pi/3*2,
            near=0.01,
            far=100
        )
        
        # Side view camera
        side_camera = CameraConfig(
            "side_camera",
            pose=look_at(eye=[0.5, 0, 0.5], target=[0, 0, 0.2]),
            width=128,
            height=128,
            fov=np.pi/3 * 2,
            near=0.01,
            far=100
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
            fov=np.pi/3,
            near=0.01,
            far=100
        )