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


@register_env("StackTower-v1", max_episode_steps=50)
class StackTowerEnv(BaseEnv):
    """
    **Task Description:**
    Stack the sphere, the shallow bin and the board on the table.

    **Randomizations:**
    - The position of the bin, the sphere and the board are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    the sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1], and the board is initialized in [-0.1, -0.05] x [-0.1, 0.1].

    **Success Conditions:**
    - The sphere is placed on the top of the bin, the bin is placed on the board, and the board is placed on the table. The robot remains static and the gripper is not closed at the end state.
    """

    #_sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    radius = 0.02  # radius of the sphere
    goal_region_radius = 0.05  # radius of the goal region
    inner_side_half_len = 0.02  # side length of the bin's inner square
    short_side_half_size = 0.0025  # length of the shortest edge of the block
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

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _build_bin(self, radius):
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
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
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        # build the dynamic bin
        return builder.build_dynamic(name="bin")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
        )

        # load the bin
        self.bin = self._build_bin(self.radius)
        
        # load the board
        self.board = actors.build_box(
            self.scene,
            half_sizes=self.board_half_size,
            color=np.array([12, 255, 160, 255]) / 255,
            name="board",
            body_type="kinematic",
        )

        # self.goal_region = actors.build_red_white_target(
        #     self.scene,
        #     radius=self.goal_region_radius,
        #     thickness=1e-5,
        #     name="goal_region",
        #     add_collision=False,
        #     body_type="kinematic",
        # )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the sphere in the first 1/4 zone along the x-axis (so that it doesn't collide the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.15 - 0.25)[
                ..., 0
            ]  # first 1/4 zone of x ([-0.25, -0.1])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[
                ..., 0
            ]  # spanning all possible ys
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # init the board in the second 1/4 zone along the x-axis (so that it doesn't collide the sphere)
            board_pos = torch.zeros((b, 3))
            board_pos[:, 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[
                ..., 0
            ]  # second 1/4 zone of x ([-0.1, -0.08])
            board_pos[:, 1] = (torch.rand((b, 1)) * 0.02 - 0.1)[..., 0]
            board_pos[:, 2] = self.board_half_size[2]
            q = [1, 0, 0, 0]
            board_pose = Pose.create_from_pq(p=board_pos, q=q)
            self.board.set_pose(board_pose)

            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the sphere)
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )  # the last 1/2 zone of x ([0, 0.1])
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # spanning all possible ys
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

            # target_region_xyz = xyz - torch.tensor([0, 0.1 + self.goal_region_radius, 0])
            # target_region_xyz[..., 2] = 1e-3
            # target_region_pose = Pose.create_from_pq(p=target_region_xyz, q=euler2quat(0, np.pi / 2, 0))
            # self.goal_region.set_pose(target_region_pose)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin.pose.p
        pos_board = self.board.pose.p
        # check if the obj is on the bin
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
        # check if the bin is on the board
        offset = pos_bin - pos_board
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.block_half_size[0]) <= 0.005
        )
        is_bin_on_board = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        if_bin_grasped = self.agent.is_grasping(self.bin)
        success = is_obj_on_bin & is_bin_on_board & is_obj_static & (~is_obj_grasped)
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_bin_grasped": if_bin_grasped,
            "is_obj_on_bin": is_obj_on_bin,
            "is_bin_on_board": is_bin_on_board,
            "is_obj_static": is_obj_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
            board_pos=self.board.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                tcp_to_bin_pos=self.bin.pose.p - self.agent.tcp.pose.p,
                tcp_to_board_pos=self.board.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        bin_pos = self.bin.pose.p
        bin_to_tcp_dist = torch.linalg.norm(tcp_pose - bin_pos, axis=1)
        # we want the agent to first move the bin to the board, then move the obj to the bin
        # so we give a higher reward for the agent to move the bin to the board
        reward = 2 * (1 - 0.4 * torch.tanh(5 * obj_to_tcp_dist) - 0.6 * torch.tanh(5 * bin_to_tcp_dist))

        # grasp and place reward
        obj_pos = self.obj.pose.p
        bin_top_pos = self.bin.pose.p.clone()
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
        obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
        board_top_pos = self.board.pose.p.clone()
        board_top_pos[:, 2] = board_top_pos[:, 2] + self.board_half_size[2] + self.block_half_size[0]
        bin_to_board_top_dist = torch.linalg.norm(board_top_pos - bin_pos, axis=1)
        # Similarly. We want the agent to first move the bin to the board, then move the obj to the bin
        # place_reward = 1 - torch.tanh(5.0 * obj_to_bin_top_dist)
        place_reward_obj = (1 - torch.tanh(5.0 * obj_to_bin_top_dist)) * 0.4
        place_reward_bin = (1 - torch.tanh(5.0 * bin_to_board_top_dist)) * 0.6
        reward[info["is_obj_grasped"]] = (2 + place_reward_obj)[info["is_obj_grasped"]]
        reward[info["is_bin_grasped"]] = (2 + place_reward_bin)[info["is_bin_grasped"]]

        # ungrasp and static reward for sphere
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_obj_grasped
        ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        robot_static_reward = self.agent.is_static(
            0.2
        )  # keep the robot static at the end state, since the sphere may spin when being placed on top
        reward[info["is_obj_on_bin"]] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[info["is_obj_on_bin"]]

        # ungrasp and static reward for bin
        is_bin_grasped = info["is_bin_grasped"]
        ungrasp_reward_bin = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward_bin[
            ~is_bin_grasped
        ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        v = torch.linalg.norm(self.bin.linear_velocity, axis=1)
        av = torch.linalg.norm(self.bin.angular_velocity, axis=1)
        static_reward_bin = 1 - torch.tanh(v * 10 + av)
        robot_static_reward_bin = self.bin.is_static(
            0.2
        )
        reward[info["is_bin_on_board"]] = (
            6 + (ungrasp_reward_bin + static_reward_bin + robot_static_reward_bin) / 3.0
        )[info["is_bin_on_board"]]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward