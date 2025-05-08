import sapien
import torch
import numpy as np
from typing import Dict, Any, Union
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig

# geometry (half-size, metre) & colours 半边长
BLOCK_SIZE = [0.025, 0.025, 0.025]      # 5 cm cube
CARD_SIZE  = [0.06,  0.09,  0.002]      # thin roof
COLORS = {
    "blue":  [0.15, 0.35, 0.9, 1.0],
    "green": [0.15, 0.8,  0.3, 1.0],
    "red":   [0.9,  0.15, 0.15, 1.0],
    "white": [1.0,  1.0,  1.0, 1.0],
}

# height checkpoints（中心坐标）
H_BLUE  = BLOCK_SIZE[2]                                           # 0.025
H_GREEN = H_BLUE  + BLOCK_SIZE[2]*2                               # 0.075
H_CARD  = H_GREEN + BLOCK_SIZE[2]   + CARD_SIZE[2]                # 0.102
H_RED   = H_CARD  + CARD_SIZE[2]    + BLOCK_SIZE[2]               # 0.129

XY_BLOCK_TOL  = BLOCK_SIZE[0] * 0.8   # 2.0 cm
XY_CARD_TOL   = CARD_SIZE[0]        # 2.5 cm   
Z_BLOCK_TOL   = BLOCK_SIZE[2] * 0.5   # 1.25 cm
Z_CARD_TOL    = CARD_SIZE[2]   # 0.2 cm


@register_env("ColorRoofStack-v1", max_episode_steps=150)
class ColorRoofStackEnv(BaseEnv):
    """
    Task: 造一个小塔
      1. 将蓝色方块放到目标位置（标黄）
      2. 把绿色方块叠在蓝色上
      3. 把白色卡片水平盖在绿色方块顶面
      4. 把红色方块放到卡片正中央
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # constructor
    def __init__(self, *args,
                 robot_uids="panda",
                 num_envs=1,
                 reconfiguration_freq=None,
                 **kwargs):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args,
                         robot_uids=robot_uids,
                         num_envs=num_envs,
                         reconfiguration_freq=reconfiguration_freq,
                         **kwargs)

    # scene & asset creation
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # three blocks
        self.blocks = {}
        for color in ["blue", "green", "red"]:
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=BLOCK_SIZE)
            builder.add_box_visual(half_size=BLOCK_SIZE,
                                   material=sapien.render.RenderMaterial(base_color=COLORS[color]))
            builder.initial_pose = sapien.Pose(p=[0, 0, BLOCK_SIZE[2]+0.3], q=[1, 0, 0, 0])
            self.blocks[color] = builder.build(name=f"{color}_block")

        # white card
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=CARD_SIZE)
        builder.add_box_visual(half_size=CARD_SIZE,
                               material=sapien.render.RenderMaterial(base_color=COLORS["white"]))
        builder.initial_pose = sapien.Pose(p=[0, 0, CARD_SIZE[2]+0.3], q=[1, 0, 0, 0])
        self.card = builder.build(name="card_roof")

        # 目标区域可视化 (半透明黄矩形)
        # 这只是一个标记，没有碰撞
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[BLOCK_SIZE[0], BLOCK_SIZE[1], 0.001],
                               material=sapien.render.RenderMaterial(base_color=[1, 1, 0, 0.3]))
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.001], q=[1, 0, 0, 0])
        self.goal_marker = builder.build_static(name="goal_marker")

    # episode init / randomisation
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        b, device = len(env_idx), self.device

        # 随机目标 xy（桌面 25 cm 方区域）
        goal = torch.zeros((b, 3), device=device)
        goal[:, :2] = (torch.rand((b, 2), device=device) - 0.5) * 0.25
        goal[:, 2]  = H_BLUE
        self.goal_position = goal                      # (b,3)
        self.goal_marker.set_pose(Pose.create_from_pq(p=goal, q=[1, 0, 0, 0]))

        # helper: 随机生成 pose
        def rand_pose(z):
            pos = torch.zeros((b, 3), device=device)
            pos[:, :2] = (torch.rand((b, 2), device=device) - 0.5) * 0.35
            pos[:, 2]  = z
            return Pose.create_from_pq(p=pos, q=[1, 0, 0, 0])

        self.blocks["blue"].set_pose(rand_pose(BLOCK_SIZE[2]))
        self.blocks["green"].set_pose(rand_pose(BLOCK_SIZE[2]))
        self.blocks["red"].set_pose(rand_pose(BLOCK_SIZE[2]))
        self.card.set_pose(rand_pose(CARD_SIZE[2]))

        # 清零速度
        for actor in [*self.blocks.values(), self.card]:
            actor.set_linear_velocity(torch.zeros((b, 3), device=device))
            actor.set_angular_velocity(torch.zeros((b, 3), device=device))

    # observations
    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose,
                   goal_pos=self.goal_position)
        if self.obs_mode_struct.use_state:
            for name, act in self.blocks.items():
                obs[f"{name}_pose"] = act.pose.raw_pose
                obs[f"{name}_vel"]  = act.linear_velocity
            obs["card_pose"] = self.card.pose.raw_pose
            obs["card_vel"]  = self.card.linear_velocity
        return obs

    # success criteria
    def evaluate(self):
        device = self.device
        b = self.num_envs
        # 收集所有方块和卡片的当前中心位置（批量 b）
        p = {k: v.pose.p for k, v in self.blocks.items()}   # {"blue":(b×3), ...}
        # p["blue"][i] = [x_i, y_i, z_i] 表示第 i 个环境里蓝色方块中心的三维坐标。
        p_card = self.card.pose.p                           # (b×3)
        goal_xy = self.goal_position[:, :2]                 # (b×2)

        # 坐标约束：平面距离 < XY_TOL 且 垂直距离在 < Z_TOL
        # 1 蓝方块到目标
        blue_ok = (torch.linalg.norm(p["blue"][:, :2] - goal_xy, dim=-1) < XY_BLOCK_TOL) & \
                  (torch.abs(p["blue"][:, 2] - H_BLUE) < Z_BLOCK_TOL)

        # 2 绿叠蓝
        green_ok = (torch.linalg.norm(p["green"][:, :2] - p["blue"][:, :2], dim=-1) < XY_BLOCK_TOL) & \
                   (torch.abs(p["green"][:, 2] - H_GREEN) < Z_BLOCK_TOL)

        # 3 卡片盖绿
        card_ok = (torch.linalg.norm(p_card[:, :2] - p["green"][:, :2], dim=-1) < XY_CARD_TOL) & \
                  (torch.abs(p_card[:, 2] - H_CARD) < Z_CARD_TOL)

        # 4 红块放卡
        red_ok = (torch.linalg.norm(p["red"][:, :2] - p_card[:, :2], dim=-1) < XY_BLOCK_TOL) & \
                 (torch.abs(p["red"][:, 2] - H_RED) < Z_BLOCK_TOL)

        # 成功条件：四项均满足
        success = blue_ok & green_ok & card_ok & red_ok
        return dict(success=success,
                    blue_at_goal=blue_ok,
                    green_on_blue=green_ok,
                    card_on_green=card_ok,
                    red_on_roof=red_ok)

    # dense reward（简单 shaping）
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        device = self.device
        reward = torch.zeros(self.num_envs, device=device)

        # 完成 +10
        reward = torch.where(info["success"], reward + 10.0, reward)

        # 距离 shaping（exp 距离）计算四段子目标的平面距离
        p = {k: v.pose.p for k, v in self.blocks.items()}
        p_card = self.card.pose.p
        goal_xy = self.goal_position[:, :2]

        dists = torch.stack([
            torch.linalg.norm(p["blue"][:, :2] - goal_xy, dim=-1),              # blue→goal
            torch.linalg.norm(p["green"][:, :2] - p["blue"][:, :2], dim=-1),    # green→blue
            torch.linalg.norm(p_card[:, :2]    - p["green"][:, :2], dim=-1),    # card→green
            torch.linalg.norm(p["red"][:, :2]  - p_card[:, :2], dim=-1),        # red→card
        ], dim=0)

        # 不同子目标的 shaping 强度
        scales = torch.tensor([0.3, 0.3, 0.2, 0.2], device=device)
        
        reward += (torch.exp(-10.0 * dists) * scales[:, None]).sum(dim=0)
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 11.0   # 最多≈10+1

    # cameras
    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig("top",
                         pose=look_at([0, 0, 0.8], [0, 0, 0]),
                         width=128, height=128, fov=np.pi/3,
                         near=0.01, far=100),
            CameraConfig("side",
                         pose=look_at([0.5, 0, 0.4], [0, 0, 0.1]),
                         width=128, height=128, fov=np.pi/3,
                         near=0.01, far=100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_cam",
                            pose=look_at([0.6, 0.6, 0.6], [0, 0, 0.15]),
                            width=512, height=512, fov=np.pi/3,
                            near=0.01, far=100)
