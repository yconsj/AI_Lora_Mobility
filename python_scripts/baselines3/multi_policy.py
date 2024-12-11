import torch as th
import torch.nn.functional as F

from stable_baselines3 import DQN, PPO
from twod_env import TwoDEnv, FrameSkip
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import MlpExtractor
import torch.nn as nn
class MultiTaskPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MultiTaskPolicy, self).__init__(*args, **kwargs)

        # Auxiliary head: Predict positions of two nodes (x1, y1, x2, y2)
        self.localization_head = nn.Sequential(
        nn.Linear(18, 64),  # Input size: 18 (match the size of aux_features)
        nn.ReLU(),
        nn.Linear(64, 4)  # Output size: 4 (for 4 predicted values such as x, y positions)
        )
        # Policy head: Use shared features + predicted positions for actions
        print(self.features_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(self.features_dim + 4, 64),  # Include node predictions
            nn.ReLU(),
            nn.Linear(64, self.action_space.n)  # Output action (e.g., 2 for x and y movement)
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False):

        # Split observation: Auxiliary task uses only the relevant part
        aux_inputs = obs[:, 5:5+(3*6)] 
        full_inputs = obs  # Entire observation for the policy

        # Shared feature extraction for both tasks
        shared_features = self.extract_features(full_inputs)

        # Auxiliary task: Predict node positions using only relevant inputs
        aux_features = self.extract_features(aux_inputs)  # Optional separate feature extractor
        print(f"Auxiliary task features shape: {aux_features.shape}")
        predicted_positions = self.localization_head(aux_features)
        print(f"Predicted positions shape: {predicted_positions.shape}")

        # Combine shared features with predicted positions
        combined_input = th.cat([shared_features, predicted_positions], dim=1)

        # Policy head: Compute actions
        mean_actions = self.policy_head(combined_input)

        # Add exploration (std deviation can be learned or fixed)
        log_std = th.ones_like(mean_actions) * -0.5  # Example: Fixed log std
        std = th.exp(log_std)
        action_distribution = th.distributions.Normal(mean_actions, std)

        # Sample actions or take deterministic ones
        if deterministic:
            actions = mean_actions
        else:
            actions = action_distribution.sample()

        # State value: Use features alone (or combined with predictions if needed)
        values = self.value_net(shared_features)

        return actions, values, predicted_positions

    def predict_positions(self, obs: th.Tensor):
        # Extract only relevant inputs for auxiliary task
        aux_inputs = obs[:, 5:5+(3*6)] 
        aux_features = self.extract_features(aux_inputs)
        return self.localization_head(aux_features)


class MultiTaskCallback(BaseCallback):
    def __init__(self, localization_loss_weight=0.1, verbose=0):
        super(MultiTaskCallback, self).__init__(verbose=verbose)
        self.localization_loss_weight = localization_loss_weight

    def _on_step(self) -> bool:
        # Access model and environment
        model = self.model
        env = self.training_env.envs[0]
        
        # Access true positions from the environment info
        for i in range(len(self.locals['obs'])):
            true_positions = th.tensor(self.locals['infos'][i]["true_node_positions"]).flatten()
            obs = th.tensor(self.locals['obs'][i], dtype=th.float32)
            
            # Predict node positions
            predicted_positions = model.policy.predict_positions(obs)
            
            # Compute localization loss (MSE)
            localization_loss = F.mse_loss(predicted_positions, true_positions)
            
            # Combine RL loss and localization loss
            model.policy.optimizer.zero_grad()
            rl_loss = self.locals['loss']
            total_loss = rl_loss + self.localization_loss_weight * localization_loss
            
            # Backpropagate the combined loss
            total_loss.backward()
            model.policy.optimizer.step()
        
        return True
    




import multiprocessing
## tensorboard --logdir ./tensorboard/;./tensorboard/  ##

def make_skipped_env():
    env = TwoDEnv(render_mode="none", timeskip=10)
    env = FrameSkip(env, skip=10)  # Frame skip for action repeat
    return env
# http://localhost:6006/

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access total_recieved from the 'infos' list
        infos = self.locals['infos']
        dones = self.locals["dones"]
        for i, done in enumerate(dones):
            if done:
                total_received_values = infos[i].get("total_received", 0)
                total_misses_values = infos[i].get('total_misses', 0)
                self.logger.record("total_received", total_received_values)
                self.logger.record("total_misses", total_misses_values)
        return True

def main():
    envs = 8
    env = make_vec_env(make_skipped_env, n_envs=envs, vec_env_cls= SubprocVecEnv)

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=20, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1, best_model_save_path="stable-model-2d-best")
    print("Learning started")

    # Instantiate the PPO model
    model = PPO(MultiTaskPolicy, env, tensorboard_log="./tensorboard/", verbose=1)

    # Train with the multitask callback
    callback = MultiTaskCallback(localization_loss_weight=0.1)
    model.learn(total_timesteps=10000, callback=[eval_callback, TensorboardCallback(), callback])

    # Save the model
    model.save("multi_task_agent")
if __name__ == '__main__':
    # Protect the entry point for multiprocessing
    multiprocessing.set_start_method('spawn')  # Ensure spawn is used on Windows
    main()