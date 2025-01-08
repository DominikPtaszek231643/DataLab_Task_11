"""
train_rl.py

Script for Task 11:
- Trains an RL model (PPO, by default) on the OT2Env environment.
- Logs training metrics to Weights & Biases (wandb).
- Optionally integrates with ClearML if desired. s
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse

import sys

# If using ClearML:
from stable_baselines3 import PPO  # or SAC, TD3, etc.
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb
# Import your custom environment
from ot2_gym_wrapper import OT2Env
from clearml import Task

# If using ClearML, uncomment and adapt the lines below:
task = Task.init(project_name="Mentor Group J/Group 1", task_name="testing_tuning_test_Dominik")
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

"""
Main function to parse arguments, initialize wandb, and train the RL agent.
"""

# --------------------
# Parse command line arguments
# --------------------
parser = argparse.ArgumentParser(description="Train PPO on OT2Env with W&B logging")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for PPO")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for PPO")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run per update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for PPO")
parser.add_argument("--total_timesteps", type=int, default=200_000, help="Total timesteps to train")
parser.add_argument("--project_name", type=str, default="Mentor Group J_Group 1", help="W&B project name")
parser.add_argument("--env_render", type=bool, default=False, help="If set, render PyBullet GUI")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

args = parser.parse_args()
task.connect(args.__dict__)  # Connect the parsed arguments as a dictionary

print(f"Connected arguments to ClearML: {args}")
# --------------------
# Initialize Weights & Biases
# --------------------
# It's highly recommended to remove the hardcoded API key for security reasons.
# Instead, retrieve it from environment variables.
# Example:
# export WANDB_API_KEY='your_secure_api_key'

wandb_api_key = '2294ea30c7144e1e29b54e608ec172f37f905c08'

if not wandb_api_key:
    print("Error: WANDB_API_KEY not found in environment variables.")
    sys.exit(1)
os.environ['WANDB_API_KEY'] = wandb_api_key

run = wandb.init(
    project=args.project_name,
    config={
        "algorithm": "PPO",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "total_timesteps": args.total_timesteps,
        "env_render": args.env_render,
        "seed": args.seed,
        "gamma": args.gamma
    },
)

# Ensure the model save directory exists
model_save_dir = f"models/{run.id}"
os.makedirs(model_save_dir, exist_ok=True)

wandb_callback = WandbCallback(
    model_save_freq=args.total_timesteps // 10,  # Save the model every 10% of total timesteps
    model_save_path=model_save_dir,
    verbose=2
)

# --------------------
# Create Environment
# --------------------
try:
    env = OT2Env(seed=args.seed, render=args.env_render, max_steps=1000, )
    env = DummyVecEnv([lambda: env])
except Exception as e:
    print(f"Error creating environment: {e}")
    sys.exit(1)

# --------------------
# Define RL Model
# --------------------
try:
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        tensorboard_log=f"runs/{run.id}",
        seed=args.seed,
        gamma=args.gamma
    )
except Exception as e:
    print(f"Error defining the model: {e}")
    sys.exit(1)

# --------------------
# Train the model
# --------------------
try:
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=wandb_callback,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name=f"runs/{run.id}"
    )
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)

# Finish wandb run
wandb.finish()

# --------------------
# Save the final model
# --------------------
final_model_path = f"{model_save_dir}/final_model"
try:
    model.save(final_model_path)
    print(f"Training completed. Final model saved at {final_model_path}")
except Exception as e:
    print(f"Error saving final model: {e}")
    sys.exit(1)

# If using ClearML, the experiment data will be uploaded automatically after this script ends.
