import argparse

import gymnasium as gym
import pygame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gym-platformer in manual or auto mode.")
    parser.add_argument(
        "--game-mode",
        choices=["manual", "auto"],
        default="manual",
        help="manual: keyboard controls, auto: random actions",
    )
    return parser.parse_args()


def get_manual_action(keys: pygame.key.ScancodeWrapper, cfg: object) -> int:
    go_left = keys[cfg.KEY_LEFT] or keys[pygame.K_LEFT] or keys[pygame.K_a]
    go_right = keys[cfg.KEY_RIGHT] or keys[pygame.K_RIGHT] or keys[pygame.K_d]
    jump = keys[cfg.KEY_UP] or keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]

    if jump and go_left:
        return 2
    if jump and go_right:
        return 3
    if jump:
        return 4
    if go_left:
        return 0
    if go_right:
        return 1
    return 5


if __name__ == "__main__":
    args = parse_args()

    env = gym.make("gym_platformer:platformer-v0", render_mode="human", ep_duration=float("inf"))
    env.reset()

    cfg = env.unwrapped.cfg

    running = True
    done = False
    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if args.game_mode == "manual":
            action = get_manual_action(keys, cfg)
        else:
            action = env.action_space.sample()

        if keys[pygame.K_ESCAPE]:
            running = False
            continue

        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
