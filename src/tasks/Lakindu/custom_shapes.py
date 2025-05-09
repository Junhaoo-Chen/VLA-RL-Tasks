# custom_shapes.py
import sapien
import numpy as np
from typing import List
import torch

def create_trash_can(scene, pose=None, name_suffix="", 
                    height = 0.3,      # Height: 40cm
                    width = 0.2,       # Width: 30cm
                    length = 0.2,      # Length: 40cm
                    thickness = 0.01,
                    density=1000,
                    color=[0.2, 0.2, 0.2, 1.0],
                    shape=False
                    ):
    """
    Creates a weighted rectangular trash can with configurable physical properties.
    
    Args:
        scene: ManiSkill scene instance
        pose: Optional initial pose
        name_suffix: Suffix for unique naming
        static_friction: Friction coefficient when not moving
        dynamic_friction: Friction coefficient when moving
        restitution: Bounciness factor (0-1)
        density: Material density in kg/m³
    """
    builder = scene.create_actor_builder()
    
    # Create the four walls
    walls = [
        # Front wall
        {"size": [length/2, thickness/2, height/2], "pos": [0, -width/2, height/2]},
        # Back wall
        {"size": [length/2, thickness/2, height/2], "pos": [0, width/2, height/2]},
        # Left wall
        {"size": [thickness/2, width/2, height/2], "pos": [-length/2, 0, height/2]},
        # Right wall
        {"size": [thickness/2, width/2, height/2], "pos": [length/2, 0, height/2]}
    ]
    
    # Add walls
    for wall in walls:
        builder.add_box_collision(
            half_size=wall["size"],
            pose=sapien.Pose(wall["pos"]),
            density=density
        )
        builder.add_box_visual(
            half_size=wall["size"],
            pose=sapien.Pose(wall["pos"]),
            material=sapien.render.RenderMaterial(
                base_color=color
            )
        )
    
    # Add bottom
    builder.add_box_collision(
        half_size=[length/2, width/2, thickness/2],
        pose=sapien.Pose([0, 0, thickness/2]),
        density=density
    )
    builder.add_box_visual(
        half_size=[length/2, width/2, thickness/2],
        pose=sapien.Pose([0, 0, thickness/2]),
        material=sapien.render.RenderMaterial(
            base_color=color
        )
    )
    
    # Add weighted base for stability
    builder.add_box_collision(
        half_size=[length/2 - thickness, width/2 - thickness, thickness/2],
        pose=sapien.Pose([0, 0, thickness/2]),
        density=density * 2,  # Double density for stability
    )
    
    # Set initial pose if provided
    if pose is not None:
        builder.initial_pose = pose
    
    # Build the actor
    unique_name = f"trash_can_{name_suffix}" if name_suffix else "trash_can_0"
    trash_can = builder.build(name=unique_name)
    
    if shape:
        if color == [0.5, 0.0, 0.0, 1.0]: trash_can.shape = "sphere" # red: sphere
        elif color == [0, 0.5, 0, 1.0]: trash_can.shape = "box" # green: box
        elif color == [0, 0, 0.5, 1.0]: trash_can.shape = "capsule"  # blue: capsule 
        elif color == [0.5, 0, 0.5, 1.0]: trash_can.shape = "cylinder" # purple: cylinder
        else: trash_can.shape = None
        trash_can.color = None
    else:
        if color == [0.5, 0.0, 0.0, 1.0]: trash_can.color = "red"
        elif color == [0, 0.5, 0, 1.0]: trash_can.color = "green"
        elif color == [0, 0, 0.5, 1.0]: trash_can.color = "blue"
        elif color == [0.5, 0, 0.5, 1.0]: trash_can.color = "purple"
        else: trash_can.color = None
        trash_can.shape = None
    return trash_can



def create_random_trash(scene, num_pieces: int = 1, 
                       min_size: float = 0.02, 
                       max_size: float = 0.08,
                       basic=False,
                       color_flag=False,
                       name_suffix="") -> List:
    """
    Creates random trash pieces with various shapes, sizes, and colors.
    
    Args:
        scene: ManiSkill scene instance
        num_pieces: Number of trash pieces to generate
        min_size: Minimum size in meters
        max_size: Maximum size in meters
    
    Returns:
        List of created trash actors
    """
    def random_color(color_flag):
        # Generate random RGB color with slight transparency
        if color_flag:
            color_idx  =int(np.random.choice([0,1,2,3]))
            idx_to_color = {
                0: [0.5, 0.0, 0.0, 1.0],  # red
                1: [0, 0.5, 0, 1.0],      # green
                2: [0, 0, 0.5, 1.0],      # blue
                3: [0.5, 0, 0.5, 1.0]     # purple
            }
            return idx_to_color[color_idx]
        else:
            return [np.random.random(), np.random.random(), np.random.random(), min(np.random.uniform(0.3, 1.7),1.0)]
    
    def random_pose_in_box(box_size):
        # Generate random position within a box volume
        pos = np.random.uniform(-box_size, box_size, 3)
        # Random rotation quaternion
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.uniform(-1, 1, 3)
        axis = axis / np.linalg.norm(axis)
        sin_a = np.sin(angle / 2)
        cos_a = np.cos(angle / 2)
        quat = [cos_a, axis[0] * sin_a, axis[1] * sin_a, axis[2] * sin_a]
        return sapien.Pose(pos, quat)

    trash_pieces = []
    
    for j in range(num_pieces):
        builder = scene.create_actor_builder()
        

        # Random size within constraints
        if basic:
            color = [0, 0, 1, 1]
            shape_type = 'sphere'
            density = 500
            size = 0.08
        else:
            # randomization
            color = random_color(color_flag)
            shape_type = np.random.choice(['sphere', 'box', 'capsule', 'cylinder'])
            density = np.random.uniform(100, 1000)
            size = np.random.uniform(min_size, max_size)

        if shape_type == 'sphere':
            radius = size / 2
            builder.add_sphere_collision(
                radius=radius,
                density=density
            )
            builder.add_sphere_visual(
                radius=radius,
                material=sapien.render.RenderMaterial(base_color=color)
            )
            
        elif shape_type == 'box':
            half_sizes = np.random.uniform(min_size/2, max_size/2, 3)
            builder.add_box_collision(
                half_size=half_sizes,
                density=density,
            )
            builder.add_box_visual(
                half_size=half_sizes,
                material=sapien.render.RenderMaterial(base_color=color)
            )
            
        elif shape_type == 'capsule':
            radius = size / 4
            half_length = size / 2
            builder.add_capsule_collision(
                radius=radius,
                half_length=half_length,
                density=density
            )
            builder.add_capsule_visual(
                radius=radius,
                half_length=half_length,
                material=sapien.render.RenderMaterial(base_color=color)
            )
            
        elif shape_type == 'cylinder':
            radius = size / 2.5
            half_length = size / 2
            builder.add_cylinder_collision(
                radius=radius,
                half_length=half_length,
                density=density
            )
            builder.add_cylinder_visual(
                radius=radius,
                half_length=half_length,
                material=sapien.render.RenderMaterial(base_color=color)
            )
        
        # Build the trash piece with random pose
        trash = builder.build(name=f"trash_piece_{name_suffix}_{j}")
        trash.set_pose(random_pose_in_box(size))
        
        shape_to_idx = {'sphere': 0, 'box': 1, 'capsule': 2, 'cylinder': 3}
        trash.shape = torch.tensor(shape_to_idx[shape_type])

        if color_flag:
            color_to_idx = {tuple([0.5, 0.0, 0.0, 1.0]): 0, tuple([0, 0.5, 0, 1.0]): 1, tuple([0, 0, 0.5, 1.0]): 2, tuple([0.5, 0, 0.5, 1.0]): 3}
            trash.color = torch.tensor(color_to_idx[tuple(color)])
        else:
            trash.color = torch.tensor(color)
        trash_pieces.append(trash)
    
    return trash_pieces

