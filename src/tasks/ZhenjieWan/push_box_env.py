from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("PushBox-v1", max_episode_steps=50)
class PushBoxEnv(BaseEnv):
    """
    **Task Description:**
    Push the green box in the bin to the goal region and you should move any barrier out of the way in order to do so.

    **Randomizations:**
    - The initial position of the target box in the bin is fixed in the upper conner of the bin
    - The initial position of the barrier in the bin is fixed in the center of the bin
    - the target goal region is marked by a red/white circular target, and its position is fixed in the downward conner of the bin
    - The initial position of the bin is randomized within a certain range

    **Success Conditions:**
    - the target's xy position is within goal_radius (default 0.05) of the target's xy position by euclidean distance and the target box is still on the table. (prevent the reward hacking)
    
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    radius = 0.02  # radius of the sphere
    goal_region_radius = 0.05  # radius of the goal region
    inner_side_half_len = 0.10  # side length of the bin's inner square
    short_side_half_size = 0.010  # length of the shortest edge of the block
    board_half_size = 0.03  # half size of the board
    board_height = 0.01  # height of the board
    board_half_size = [
        board_half_size,
        board_half_size,
        board_height,
    ]
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one
    padding_block_half_size = [
        inner_side_half_len / 3,
        inner_side_half_len / 3,
        2 * short_side_half_size,
    ]
    box_half_size = [
        padding_block_half_size[0],
        padding_block_half_size[1],
        padding_block_half_size[2] * 1.5,
    ]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    def _build_bin(self, radius):
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        dx_padding = self.inner_side_half_len - self.padding_block_half_size[0]
        dy_padding = self.inner_side_half_len - self.padding_block_half_size[1]
        dz_padding = self.edge_block_half_size[2] + self.block_half_size[0]
        # build the bin bottom and edge blocks
        poses = [
            # bottom block
            sapien.Pose([0, 0, 0]),

            # edge blocks
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]

        padding_poses = [
            # padding blocks
            sapien.Pose([-dx_padding, -dy_padding, dz_padding]),
            sapien.Pose([-dx_padding, dy_padding, dz_padding]),
            sapien.Pose([dx_padding, -dy_padding, dz_padding]),
            sapien.Pose([dx_padding, dy_padding, dz_padding]),
        ]

        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]

        padding_half_sizes = [
            self.padding_block_half_size,
            self.padding_block_half_size,
            self.padding_block_half_size,
            self.padding_block_half_size,
        ]

        color = np.array([12, 42, 160, 255]) / 255
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=sapien.render.RenderMaterial(base_color=color))
        
        for pose, half_size in zip(padding_poses, padding_half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=sapien.render.RenderMaterial(base_color=color))        

        # build the dynamic bin
        return builder.build_kinematic(name="bin")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the bin
        self.bin = self._build_bin(self.radius)
        
        self.target_box = actors.build_box(
            self.scene,
            half_sizes=self.box_half_size,
            color=np.array([12, 255, 160, 255]) / 255,
            name="target_box",
            body_type="dynamic",
        )

        self.barrier = actors.build_box(
            self.scene,
            half_sizes=self.box_half_size,
            color=np.array([12, 160, 255, 255]) / 255,
            name="barrier",
            body_type="dynamic",
        )
    
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.box_half_size[0],
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # spanning all possible xs
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # spanning all possible ys
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

           # init the target box in the bin
            target_box_pos = [
                pos[:, 0] - (self.inner_side_half_len - self.padding_block_half_size[1]),
                pos[:, 1],
                pos[:, 2] + self.box_half_size[2] + self.block_half_size[0],
            ]
            target_box_pose = Pose.create_from_pq(p=target_box_pos, q=q)
            self.target_box.set_pose(target_box_pose)
            self.origin_target_box_pos = target_box_pose.p

            # init the barrier in the bin
            barrier_pos = [
                pos[:, 0],
                pos[:, 1],
                pos[:, 2] + self.box_half_size[2] + self.block_half_size[0],
            ]
            barrier_pose = Pose.create_from_pq(p=barrier_pos, q=q)
            self.barrier.set_pose(barrier_pose)
            self.origin_barrier_pos = barrier_pose.p

            # init the goal region
            goal_region_pos = [
                pos[:, 0] + (self.inner_side_half_len - self.padding_block_half_size[1]),
                pos[:, 1],
                pos[:, 2] + self.block_half_size[0],
            ]
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=goal_region_pos,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        
        is_target_box_placed = (
            torch.linalg.norm(
                self.target_box.pose.p[..., :2] - self.goal_region.pose.p[..., :2], dim=1
            ) 
            < self.goal_region_radius
        )

        return {
            "success": is_target_box_placed,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # if the observation mode requests to use state, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.goal_region.pose.p,
                target_box_pos=self.target_box.pose.raw_pose,
                barrier_pos=self.barrier.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        # Create a pose marking where the robot should push the target from that is easiest (pushing from behind the cube)
        tcp_push_pose_for_target_box = Pose.create_from_pq(
            p=self.target_box.pose.p
            + torch.tensor([-self.box_half_size[0] - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose_for_target_box = tcp_push_pose_for_target_box.p - self.agent.tcp.pose.p
        tcp_to_push_pose_for_target_box_dist = torch.linalg.norm(tcp_to_push_pose_for_target_box, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_for_target_box_dist)
        reward = reaching_reward


        # compute a placement reward to encourage robot to move the target_box to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        # Compute a z reward to encourage the robot to keep the cube on the table
        
        reached_target_box = tcp_to_push_pose_for_target_box_dist < 0.01
        target_box_to_goal_dist = torch.linalg.norm(
            self.target_box.pose.p[..., :2] - self.goal_region.pose.p[..., :2], dim=1
        )
        place_reward = 1 - torch.tanh(5 * target_box_to_goal_dist)

        # Similarly, we need to reward the robot for removing the barrier out of the way (pushing it to the left or right)
        tcp_push_pose_for_barrier = [
            Pose.create_from_pq(
                p=self.barrier.pose.p
                + torch.tensor([0, -self.box_half_size[1] - 0.005, 0], device=self.device)
            ),
            Pose.create_from_pq(
                p=self.barrier.pose.p
                + torch.tensor([0, self.box_half_size[1] + 0.005, 0], device=self.device)
            ),
        ]
        tcp_to_push_pose_for_barrier = torch.stack(
            [
                tcp_push_pose_for_barrier[0].p - self.agent.tcp.pose.p,
                tcp_push_pose_for_barrier[1].p - self.agent.tcp.pose.p,
            ],
            dim=1,
        )
        tcp_to_push_pose_for_barrier_dist = torch.linalg.norm(
            tcp_to_push_pose_for_barrier, dim=2
        )

        reward += torch.max(
            1 - torch.tanh(5 * tcp_to_push_pose_for_barrier_dist), dim=1
        ).values

        # compute the move away reward to encourage the robot to move the barrier from the original position to the left or right
        # we further multiply the move away reward by a mask reached so we only add the move away reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.

        reached_barrier = torch.logical_or(
            tcp_to_push_pose_for_barrier_dist[:, 0] < 0.01,
            tcp_to_push_pose_for_barrier_dist[:, 1] < 0.01,
        )
        left_pos_for_barrier = self.origin_barrier_pos - torch.tensor(
            [0, self.inner_side_half_len - self.padding_block_half_size[1], 0],
            device=self.device,
        )
        right_pos_for_barrier = self.origin_barrier_pos + torch.tensor(
            [0, self.inner_side_half_len - self.padding_block_half_size[1], 0],
            device=self.device,
        )
        left_dist = torch.linalg.norm(
            self.barrier.pose.p[..., :2] - left_pos_for_barrier[..., :2], dim=1
        )
        right_dist = torch.linalg.norm(
            self.barrier.pose.p[..., :2] - right_pos_for_barrier[..., :2], dim=1
        )
        move_away_reward = 1 - torch.tanh(5 * torch.min(left_dist, right_dist))

        reward += place_reward * reached_target_box + move_away_reward * reached_barrier   

        # compute the z reward to encourage the robot to keep the target_box on the table  
        desired_target_box_z = self.box_half_size[2] + 2 * self.block_half_size[0]
        current_target_box_z = self.target_box.pose.p[..., 2]
        z_deviation = torch.abs(current_target_box_z - desired_target_box_z)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        # We multiply the z reward by the place_reward and reached mask so that 
        #   we only add the z reward if the robot has reached the desired push pose
        #   and the z reward becomes more important as the robot gets closer to the goal.
        reward += place_reward * z_reward * reached_target_box

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )  