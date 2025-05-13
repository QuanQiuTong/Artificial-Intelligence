import pygame
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    NoopResetEnv
)

def make_env():
    """Create environment for manual play"""
    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=1)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env

def manual_play():
    """Main function for manual play"""
    # Initialize pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode((84, 84))
    pygame.display.set_caption("Manual Play: MsPacman")
    clock = pygame.time.Clock()

    # Create environment
    env = make_env()
    action_meanings = env.unwrapped.get_action_meanings()
    print("Action meanings:", action_meanings)

    # Print controls
    print("=== Manual Play: MsPacman ===")
    print("Controls:")
    print("  W: UP")
    print("  A: LEFT")
    print("  S: DOWN")
    print("  D: RIGHT")
    print("  ESC: Quit")
    print("\nGame starting... Use WASD to control Pac-Man!")

    key_to_action = {
        pygame.K_w: 1,
        pygame.K_d: 2,
        pygame.K_a: 3,
        pygame.K_s: 4
    }

    # Start the game
    obs, _ = env.reset()
    total_reward = 0
    episode = 1

    running = True
    while running:
        # 非阻塞事件处理
        for event in pygame.event.get(pygame.QUIT):
            running = False
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key == pygame.K_ESCAPE:
                running = False

        # 持续检测按键状态
        keys = pygame.key.get_pressed()
        action = 0
        for key, act in key_to_action.items():
            if keys[key]:
                action = act
                break

        # Take action in the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display current score
        print(f"\rEpisode: {episode} | Score: {total_reward}", end="")

        obs = next_obs

        if terminated or truncated:
            print(f"\nEpisode {episode} finished with score: {total_reward}")
            episode += 1
            total_reward = 0
            obs, _ = env.reset()
            print("\nNew episode starting...")

        
        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    manual_play()