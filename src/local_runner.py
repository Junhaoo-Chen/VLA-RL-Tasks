import os
import gymnasium as gym
import time
from mani_skill.utils.wrappers.record import RecordEpisode
# TODO: Change this to import your env
# from card_stack_env import CardStackEnv
from tasks.xuqiyue.insert_key import *

def generate_videos(n_episodes=10, max_steps_per_episode=100, video_dir="videos"):
    """
    Generate and save videos of random agent interactions in the CardStack environment.
    """
    # TODO: Change this to make your env
    env = gym.make("InsertKey-v1", obs_mode="state", render_mode="rgb_array")
    video_dir = os.path.join(video_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(video_dir, exist_ok=True)

    env = RecordEpisode(env, output_dir=video_dir, save_video=True, 
                        trajectory_name="random_actions", max_steps_per_video=max_steps_per_episode)
    for _ in range(n_episodes):
        obs, info = env.reset()
        for _ in range(max_steps_per_episode):
            action = env.action_space.sample()  # Take random action
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    env.close()

if __name__ == "__main__":
    generate_videos(n_episodes=10)