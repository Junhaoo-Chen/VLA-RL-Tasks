# artist_marker_cup_stand.py
import sapien
import sapien.core
import torch
import numpy as np
from typing import Dict, Any, Union, List
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building.actors.ycb import get_ycb_builder

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



def load_ycb_objects(scene, obj_id, count):
    """
    Load YCB objects into a scene. Also extract and store the dimensions of the loaded objects.
    Args:
        scene: ManiSkill scene object.
        obj_id: object ID to load.
        count: number of objects to render.
    Returns:
        List of built YCB objects.
    """
    ycb_objects = []

    for i in range(count):
        builder = get_ycb_builder(scene, id=obj_id, add_collision=True, add_visual=True)
        builder.initial_pose = sapien.Pose([0, 0, 0.5])  # Adjust position as needed
        obj = builder.build(name=f"ycb_{obj_id}_{i}")
        ycb_objects.append(obj)
        entity = obj._objs[0]

        # Find the RenderBodyComponent, extract the dimensions and store them
        for comp in entity.components:
            if "RenderBodyComponent" in str(type(comp)):
                aabb = comp.compute_global_aabb_tight()
                min_corner, max_corner = aabb
                dims = max_corner - min_corner
        obj.dims = dims
    return ycb_objects


@register_env("ArtistMarkerCupStand-v1", max_episode_steps=100)
class ArtistMarkerCupStand(BaseEnv):
    """
    Task: Stand the marker upright inside the red cup.
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
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
                    
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        '''
        Loads all objects like actors and articulations into the scene. Called by self._reconfigure. Given options argument is the same options dictionary passed to the self.reset function
        
        This function creates the table and the following objects:
        1. Three Marbles
        2. Two small blue blocks
        3. One yellow cup
        4. One red cup
        5. One big wooden block
        6. One marker
        7. One bowl
        8. One plate
        9. One scissor
        '''
        # Create table scene
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        # Create objects
        self.marbles = load_ycb_objects(self.table_scene.env.scene, '063-b_marbles', 3)
        self.blocks = load_ycb_objects(self.table_scene.env.scene, '070-b_colored_wood_blocks', 2)
        self.blocks.append(load_ycb_objects(self.table_scene.env.scene, '036_wood_block', 1)[0])
        self.cups = load_ycb_objects(self.table_scene.env.scene, '065-d_cups', 1) # yellow cup
        self.cups.append(load_ycb_objects(self.table_scene.env.scene, '065-e_cups', 1)[0]) # red cup
        self.plate = load_ycb_objects(self.table_scene.env.scene, '029_plate', 1)
        self.bowl = load_ycb_objects(self.table_scene.env.scene, '024_bowl', 1)
        self.scissors = load_ycb_objects(self.table_scene.env.scene, '037_scissors', 1)
        self.marker = load_ycb_objects(self.table_scene.env.scene, '040_large_marker', 1)
        ## self.cat = load_ycb_objects(self.table_scene.env.scene, '999_cute_cat', 1) 

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        '''Initialize the episode, e.g., poses of actors and articulations, as well as task relevant data like randomizing goal positions
        
        This function intializes the table and the following objects:
        1. Three Marbles
        2. Two small blue blocks
        3. One yellow cup
        4. One red cup
        5. One big wooden block
        6. One marker
        7. One bowl
        8. One plate
        9. One scissor
        '''
        # Initialize table scene
        self.table_scene.initialize(env_idx)
        
        # Generate batch dimension based on env_idx
        with torch.device(self.device):
            b = len(env_idx)
            
            # render marbles
            for obj in self.marbles:
                x = np.random.uniform(0.025, 0.05)   
                y = np.random.uniform(0.15, 0.25)   
                z = 0.01                          

                # Random rotation
                angle = np.random.uniform(0, 2 * np.pi)
                axis = np.random.uniform(-1, 1, 3)
                axis = axis / np.linalg.norm(axis)
                sin_a = np.sin(angle / 2)
                cos_a = np.cos(angle / 2)
                quat = [cos_a, axis[0] * sin_a, axis[1] * sin_a, axis[2] * sin_a]

                obj.set_pose(sapien.Pose([x, y, z], quat))
        
            # render blocks
            for i in range(len(self.blocks)):
                obj = self.blocks[i]
                x = np.random.uniform(0, 0.10)    
                y = np.random.uniform(-0.05, 0.05)   
                z = 0.04                          

                # Random rotation
                angle = np.random.uniform(0, 2 * np.pi)
                axis = np.random.uniform(-1, 1, 3)
                axis = axis / np.linalg.norm(axis)
                sin_a = np.sin(angle / 2)
                cos_a = np.cos(angle / 2)
                quat = [cos_a, axis[0] * sin_a, axis[1] * sin_a, axis[2] * sin_a]
                obj.set_pose(sapien.Pose([x, y, z], quat))
            
                if i == 1: 
                    self.blocks[-1].set_pose(sapien.Pose([x+0.1, y, z]))
                    break

            # render cups
            x = np.random.uniform(0, 0.10)    
            y = np.random.uniform(-0.30, -0.20)   
            z = 0.04                           
            self.cups[0].set_pose(sapien.Pose([x, y, z]))
            self.cups[1].set_pose(sapien.Pose([x, y-0.1, z]))

            # render bowl
            x = np.random.uniform(-0.20, -0.10)    
            y = np.random.uniform(-0.30, -0.20)   
            z = 0.04                           
            self.bowl[0].set_pose(sapien.Pose([x, y, z]))


            # render plate
            x = np.random.uniform(-0.20, -0.10)    
            y = np.random.uniform(0.05, 0.10)   
            z = 0.04                           
            self.plate[0].set_pose(sapien.Pose([x, y, z]))

            # render marker
            x = np.random.uniform(-0.40, -0.35)    
            y = np.random.uniform(0.05, 0.10)   
            z = 0.04                           
            self.marker[0].set_pose(sapien.Pose([x, y, z]))

            # render scissors
            x = np.random.uniform(-0.40, -0.35)    
            y = np.random.uniform(0.15, 0.20)   
            z = 0.04                           
            self.scissors[0].set_pose(sapien.Pose([x, y, z]))

            # Initialise at beginning of scene
            self.red_cup_radius = None
       
        

    def evaluate(self):
        """Evaluate the completion status of the marker upright task.
        
        This function checks whether the marker in the red cup and is placed perfectly upright. 

        Evaluation Criteria:
        1. Distance from centre of the red cup
            - The marker has to be within the radius of the red cup
        2. Quaterion of the marker
            - The marker has to be upright
        """
        with torch.device(self.device):
            
            # Check whether marker is in the red cup - A
            # Check whether marker is upright - B
            # Success if both
            # Rewards:
                # A - 
                    # Distance from centre of red cup
                    # Whether in red cup
                # B - 
                    # whether upright

            # Find position of the marker and red cup
            marker_pos = torch.stack([obj.pose.p for obj in self.marker])
            red_cup_pos = torch.stack([obj.pose.p for obj in self.cups[1:]])        
            
            # Calculating red cup radius
            if self.red_cup_radius is None: 
                radius_range = (self.cups[1].dims[0], self.cups[1].dims[1])
                self.red_cup_radius = max(radius_range)
                

            # Find the distance between the objects. I am assuming the x,y correspond to the centre of those objects for now
            marker_red_cup_dist = torch.sum((marker_pos[:, :, :2] - red_cup_pos[:, :, :2])**2, dim=-1)**0.5
            
            # Find whether the marker is in the red cup. The marker is in the red cup if the distance is less than max radius.
            whether_object_on_cup = (marker_red_cup_dist < self.red_cup_radius)
            
            # Check whether marker is upright
            marker_quat = torch.stack([obj.pose.q for obj in self.marker])
            
            w, x, y, z = marker_quat.unbind(-1)  # # marker_quat: (1, 1, 4), format [w, x, y, z]
            alignment = 1 - 2 * (x ** 2 + y ** 2)  # (1, 1)
            whether_marker_upright = alignment > 0.95  # (1, 1) boolean

            success = (whether_marker_upright & whether_object_on_cup).squeeze(0)
            

        return {"success": success,
                # "whether_marker_upright": whether_marker_upright,
                # "whether_object_on_cup": whether_object_on_cup,
                "marker_red_cup_dist": marker_red_cup_dist,
                "alignment": alignment
                }
        
    def _get_obs_extra(self, info: Dict):
        """Additional observations for solving the task"""
        return {}
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose
        )
        if self.obs_mode_struct.use_state:
            for i, obj in enumerate(self.objects):
                obs[f"trash_{i}_pose"] = obj.pose.raw_pose
                obs[f"trash_{i}_vel"] = obj.linear_velocity
        return obs
        
        
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute a dense reward signal to guide learning
        
        This function computes the dense reward for the marker upright task.

        Rewards: 
        1. Distance reward:
            - Distance reward for distance between marker and red cup. 
            - Closer to the cup higher the reward.
            - Only valid if marker not in cup
        2. Marker upright reward:
            - Reward if marker is upright.
            - Smaller than other rewards since this is much easier.
        3. Success reward:
            - Gives maximum possible reward if task is completed
        """
        success = info['success']
        marker_red_cup_dist = info['marker_red_cup_dist']
        alignment = info['alignment']

        # table_centre = self.table_scene.table.pose.p 


        # Distance reward for distance between marker and red cup
        distances_reward = torch.exp(-10 * marker_red_cup_dist.squeeze(0)) * 4 ## adjusted coef

        # Marker upright reward
        marker_upright_reward = torch.exp(-10 * (1 - alignment).squeeze(0))   

        # Max reward
        self.max_reward = 1*4 + 1
        
        # Success reward and final reward
        total_reward = self.max_reward if success else distances_reward + marker_upright_reward

        return total_reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward
        This function computes the normalized dense reward for the marker upright task.
        """
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