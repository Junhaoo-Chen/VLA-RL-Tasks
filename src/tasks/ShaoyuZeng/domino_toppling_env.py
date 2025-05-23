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


@register_env("DominoToppling-v1", max_episode_steps=400)
class DominoTopplingEnv(BaseEnv):
    """
    **Task Description:**
    Arrange two blue dominoes between a fixed red domino and a randomized green domino, then trigger a chain reaction by knocking down the red domino so that all dominoes fall in sequence.

    **Instructions:**
    "Arrange dominoes and then topple them down from the red one."

    **Randomization:**
    The green (end) domino's position is in a circle sector, whose center is red (start) domino and radius is randomized between [0.12, 0.15] and its z-rotation is randomized between [0, 2π].

    **Success Condition:**
    All dominoes fall in the correct sequence after arrangement. No domino should be pushed down before the arrangement is complete.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent = Union[Panda]

    # Use half of the previous values for half_size
    domino_half_size = [0.005, 0.015, 0.03]  # x, y, z half-size
    num_dominoes = 4  # red, blue1, blue2, green

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        # Table
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.dominoes = []

        # Red domino (start)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=self.domino_half_size)
        builder.add_box_visual(
            half_size=self.domino_half_size,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.03], q=[1, 0, 0, 0])
        self.red_domino = builder.build_dynamic(name="red_domino")
        self.dominoes.append(self.red_domino)

        # Green domino (end)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=self.domino_half_size)
        builder.add_box_visual(
            half_size=self.domino_half_size,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        # Place at default, will randomize in _initialize_episode
        builder.initial_pose = sapien.Pose(p=[0.1, 0.1, 0.03], q=[1, 0, 0, 0])
        self.green_domino = builder.build_dynamic(name="green_domino")
        self.dominoes.append(self.green_domino)

        # Blue dominoes (laying down, to be arranged)
        blue_positions = [[0, 0.5, 0.045], [0, 0.9, 0.045]]  # z=0.045 (center, laying flat)
        blue_dominoes = []
        for i, pos in enumerate(blue_positions):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=self.domino_half_size)
            builder.add_box_visual(
                half_size=self.domino_half_size,
                material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),
            )
            # 90 deg rotation about y axis: q = [cos(pi/4), 0, sin(pi/4), 0]
            qy = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]
            builder.initial_pose = sapien.Pose(p=pos, q=qy)
            domino = builder.build_dynamic(name=f"blue_domino_{i+1}")
            blue_dominoes.append(domino)
        self.blue_dominoes = blue_dominoes
        self.dominoes.extend(blue_dominoes)

        # For convenience
        self.domino_names = ["red_domino", "green_domino", "blue_domino_1", "blue_domino_2"]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)
            # Red domino
            self.red_domino.set_pose(Pose.create_from_pq(
                p=torch.tensor([[0, 0, 0.03]]*b, device=self.device),
                q=torch.tensor([[1, 0, 0, 0]]*b, device=self.device)
            ))
            self.red_domino.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.red_domino.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Green domino (randomized)
            r = torch.rand(b, device=self.device) * (0.15 - 0.12) + 0.12
            theta = torch.rand(b, device=self.device) * (2*math.pi/3 - math.pi/3) + math.pi/3
            x = r * torch.sin(theta)
            y = r * torch.cos(theta)
            z = torch.full((b,), 0.03, device=self.device)
            qz_angle = torch.rand(b, device=self.device) * 2 * math.pi
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = torch.cos(qz_angle/2)
            q[:, 3] = torch.sin(qz_angle/2)
            self.green_domino.set_pose(Pose.create_from_pq(
                p=torch.stack([x, y, z], dim=-1),
                q=q
            ))
            self.green_domino.set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.green_domino.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Blue dominoes (laying down)
            blue_positions = torch.tensor([[0, 0.2, 0.045], [0, 0.3, 0.045]], device=self.device)
            qy = torch.tensor([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0], device=self.device)
            for i, domino in enumerate(self.blue_dominoes):
                p = blue_positions[i].repeat(b, 1)
                q = qy.repeat(b, 1)
                domino.set_pose(Pose.create_from_pq(p=p, q=q))
                domino.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                domino.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Stage and state tracking
            self.stage = torch.zeros(b, dtype=torch.long, device=self.device)  # 0: arrange, 1: ready, 2: chain
            self.fallen_flags = torch.zeros((b, self.num_dominoes), dtype=torch.bool, device=self.device)
            self.touched_flags = torch.zeros((b, self.num_dominoes), dtype=torch.bool, device=self.device)
            self.chain_sequence = torch.zeros((b, self.num_dominoes), dtype=torch.bool, device=self.device)
            self.done = torch.zeros(b, dtype=torch.bool, device=self.device)

    def _is_domino_fallen(self, domino):
        # Returns (b,) bool tensor: True if fallen (center z < 0.015)
        z = domino.pose.p[..., 2]
        return z < 0.015

    def _is_domino_upright(self, domino):
        # Returns (b,) bool tensor: True if upright (center z ~0.03, tolerance 0.005)
        z = domino.pose.p[..., 2]
        return (z > 0.03 - 0.005) & (z < 0.03 + 0.005)

    def _is_domino_still(self, domino):
        # Returns (b,) bool tensor: True if linear velocity < 0.001
        v = domino.get_linear_velocity()
        return torch.norm(v, dim=-1) < 0.001

    def _is_domino_touched(self, domino):
        # Use is_near: check if robot TCP is within 0.02m of domino center
        tcp_pos = self.agent.tcp.pose.p  # shape (b, 3)
        domino_pos = domino.pose.p       # shape (b, 3)
        dist = torch.norm(tcp_pos - domino_pos, dim=-1)
        return dist < 0.02

    def _update_stage_and_flags(self):
        # Update fallen_flags, touched_flags, and stage transitions
        b = self.num_envs
        fallen = torch.stack([
            self._is_domino_fallen(self.red_domino),
            self._is_domino_fallen(self.green_domino),
            self._is_domino_fallen(self.blue_dominoes[0]),
            self._is_domino_fallen(self.blue_dominoes[1]),
        ], dim=-1)
        upright = torch.stack([
            self._is_domino_upright(self.red_domino),
            self._is_domino_upright(self.green_domino),
            self._is_domino_upright(self.blue_dominoes[0]),
            self._is_domino_upright(self.blue_dominoes[1]),
        ], dim=-1)
        still = torch.stack([
            self._is_domino_still(self.red_domino),
            self._is_domino_still(self.green_domino),
            self._is_domino_still(self.blue_dominoes[0]),
            self._is_domino_still(self.blue_dominoes[1]),
        ], dim=-1)
        # Touched flags (not implemented, always False)
        touched = torch.stack([
            self._is_domino_touched(self.red_domino),
            self._is_domino_touched(self.green_domino),
            self._is_domino_touched(self.blue_dominoes[0]),
            self._is_domino_touched(self.blue_dominoes[1]),
        ], dim=-1)

        # Stage transitions
        # Stage 0: arranging blue dominoes
        # If any red or green domino falls or is touched, fail
        fail0 = (fallen[:, 0] | fallen[:, 1]) | (touched[:, 0] | touched[:, 1])
        # If both blue dominoes are upright and all dominoes are still, go to stage 1
        ready = upright[:, 2] & upright[:, 3] & still.all(dim=-1)
        # Stage 1: ready to push
        # If green domino is touched, fail
        fail1 = touched[:, 1]
        # If red domino falls, go to stage 2 (chain reaction)
        chain = fallen[:, 0]
        # Stage 2: chain reaction
        # If robot touches any domino, fail
        fail2 = touched.any(dim=-1)
        # If all dominoes fall in sequence (red, blue, green), success

        for i in range(b):
            if self.stage[i] == 0:
                if fail0[i]:
                    self.done[i] = True
                elif ready[i]:
                    self.stage[i] = 1
            elif self.stage[i] == 1:
                if fail1[i]:
                    self.done[i] = True
                elif chain[i]:
                    self.stage[i] = 2
            elif self.stage[i] == 2:
                if fail2[i]:
                    self.done[i] = True

        self.fallen_flags = fallen
        self.touched_flags = touched

    def _check_chain_sequence(self):
        # Returns (b,) bool tensor: True if red falls, then blue(s), then green falls
        # For simplicity, check all fallen at end, and red falls before green
        red_fallen = self.fallen_flags[:, 0]
        blue_fallen = self.fallen_flags[:, 2] & self.fallen_flags[:, 3]
        green_fallen = self.fallen_flags[:, 1]
        # Optionally, could check timestamps/order if available
        return red_fallen & blue_fallen & green_fallen

    def evaluate(self):
        """
        Evaluate the task progress and success.

        Returns:
            - success: bool, whether the task is completed successfully.
            - fail: bool, whether the task has failed due to illegal contact.
            - stage: int, current stage (0: arranging, 1: ready, 2: chain).
            - fallen_flags: (b, 4) bool, which dominoes have fallen.
            - touched_flags: (b, 4) bool, which dominoes have been touched by robot.
            - done: bool, whether the episode should terminate.
        """
        self._update_stage_and_flags()
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Stage-based fail logic
        # Stage 0: fail if red or green domino touched
        fail_stage0 = (self.stage == 0) & (self.touched_flags[:, 0] | self.touched_flags[:, 1])
        # Stage 1: fail if green domino touched
        fail_stage1 = (self.stage == 1) & self.touched_flags[:, 1]
        fail = fail_stage0 | fail_stage1

        if (self.stage == 2).any():
            # Only succeed if chain sequence is satisfied and no domino is touched
            chain_ok = self._check_chain_sequence()
            not_touched = ~self.touched_flags.any(dim=-1)
            success = chain_ok & not_touched
            self.done = self.done | success
        self.done = self.done | fail
        return {
            "success": success,
            "fail": fail,
            "stage": self.stage.clone(),
            "fallen_flags": self.fallen_flags.clone(),
            "touched_flags": self.touched_flags.clone(),
            "done": self.done.clone(),
        }

    def _get_obs_extra(self, info: Dict):
        """
        Additional observations for solving the task, including domino poses, velocities, and stage/state info.
        """
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            stage=self.stage.clone(),
            fallen_flags=self.fallen_flags.clone(),
            touched_flags=self.touched_flags.clone(),
        )
        if self.obs_mode_struct.use_state:
            obs["red_domino_pose"] = self.red_domino.pose.raw_pose
            obs["green_domino_pose"] = self.green_domino.pose.raw_pose
            obs["blue_domino_1_pose"] = self.blue_dominoes[0].pose.raw_pose
            obs["blue_domino_2_pose"] = self.blue_dominoes[1].pose.raw_pose
            obs["red_domino_vel"] = self.red_domino.get_linear_velocity()
            obs["green_domino_vel"] = self.green_domino.get_linear_velocity()
            obs["blue_domino_1_vel"] = self.blue_dominoes[0].get_linear_velocity()
            obs["blue_domino_2_vel"] = self.blue_dominoes[1].get_linear_velocity()
        return obs

    # Dense reward function not necessary
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return torch.zeros(self.num_envs, device=self.device)

    @property
    def _default_sensor_configs(self):
        """Configure camera sensors for the environment"""
        top_camera = CameraConfig(
            "top_camera",
            pose=look_at(eye=[0, 0, 0.8], target=[0, 0, 0]),
            width=128,
            height=128,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
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
