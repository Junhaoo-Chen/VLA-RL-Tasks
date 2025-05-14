from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose


@register_env("TableScene-v1", max_episode_steps=100)
class TableSceneEnv(BaseEnv):
    """
    Simplified table scene environment with various objects
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
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # Load various objects
        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{100129}"
        )
        builder.initial_pose = sapien.Pose(p=[0.25, 0.25, 0.01])
        builder.build(name="box")

        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{4064}"
        )
        builder.initial_pose = sapien.Pose(p=[0.1, 0.4, 0.2])
        builder.build(name="bottle")

        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{101685}"
        )
        builder.initial_pose = sapien.Pose(p=[0.5, 0.3, 0.3])
        builder.build(name="pen")

        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{12727}"
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        builder.build(name="keyboard")

        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{102290}"
        )
        builder.initial_pose = sapien.Pose(p=[0.65, 0.75, 0.5])
        builder.build(name="mouse")

        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{4533}"
        )
        builder.initial_pose = sapien.Pose(p=[0.6, 0.35, 0.5])
        builder.build(name="display")

        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{179}"
        )
        builder.initial_pose = sapien.Pose(p=[0.65, 0.35, 0])
        builder.build(name="chair")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

    def evaluate(self):
        """Empty evaluation since this is just a scene"""
        with torch.device(self.device):
            return {
                "success": torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            }

    def _get_obs_extra(self, info: Dict):
        """Additional observations"""
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Empty reward since this is just a scene"""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Empty reward since this is just a scene"""
        return torch.zeros(self.num_envs, device=self.device)

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
            pose=look_at(eye=[1, 1, 0.8], target=[0, 0, 0.1]),
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )
