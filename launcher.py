import subprocess

configs = [
    # ["python", "/Users/lperelli/rl-mujoco/training.py"],
    # ["python", "/Users/lperelli/rl-mujoco/training.py"],
    # ["python", "/Users/lperelli/rl-mujoco/training.py"],
    # ["python", "/Users/lperelli/rl-mujoco/training.py"],
    [
        "python",
        "/Users/lperelli/rl-mujoco/training.py",
        "--config_path",
        "/Users/lperelli/rl-mujoco/configs/config_cheetah.yaml",
    ],
    [
        "python",
        "/Users/lperelli/rl-mujoco/training.py",
        "--config_path",
        "/Users/lperelli/rl-mujoco/configs/config_cheetah.yaml",
    ],
    # [
    #     "python",
    #     "/Users/lperelli/rl-mujoco/training.py",
    #     "--config_path",
    #     "/Users/lperelli/rl-mujoco/configs/config_cheetah.yaml",
    # ],
    # [
    #     "python",
    #     "/Users/lperelli/rl-mujoco/training.py",
    #     "--config_path",
    #     "/Users/lperelli/rl-mujoco/configs/config_cheetah.yaml",
    # ],
    # [
    #     "python",
    #     "/Users/lperelli/rl-mujoco/training.py",
    #     "--config_path",
    #     "/Users/lperelli/rl-mujoco/configs/config_cheetah.yaml",
    # ],
]

for config in configs:
    subprocess.run(config)
