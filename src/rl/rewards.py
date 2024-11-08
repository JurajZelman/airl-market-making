"""Reward networks and reward functions."""

import gymnasium as gym
import torch as th
from imitation.rewards.reward_nets import RewardNet
from stable_baselines3.common import preprocessing


class NegativeRewardNet(RewardNet):
    """
    Simple reward neural network (multi-layer perceptron) that ensures that the
    reward is always negative. This is needed to ensure that the reward that is
    passed to the AIRL discriminator makes sense from the theoretical
    perspective.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        **kwargs,
    ) -> None:
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: Indicates whether the current state should be included as
                an input to the network.
            use_action: Indicates whether the current action should be included
                as an input to the network.
            use_next_state: Indicates whether the next state should be included
                as an input to the network.
            use_done: Indicates whether the done flag should be included as an
                input to the network.
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)

        # Compute the size of the input layer
        combined_size = 0
        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(
                observation_space
            )

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(
                observation_space
            )

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        # Define the layers
        self.relu = th.nn.ReLU()
        self.log_sigmoid = th.nn.LogSigmoid()
        self.linear1 = th.nn.Linear(combined_size, 256)
        self.linear2 = th.nn.Linear(256, 256)
        self.linear3 = th.nn.Linear(256, 1)
        self.squeeze = SqueezeLayer()

        self.scale = False

    def forward(self, state, action, next_state, done) -> th.Tensor:
        """
        Forward pass of the reward network.

        Args:
            state: State of the environment.
            action: Action taken in the environment.
            next_state: Next state of the environment.
            done: Whether the episode has terminated.

        Returns:
            The reward for the given state-action-next_state-done pair.
        """
        # Concatenate the inputs
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))
        inputs_concat = th.cat(inputs, dim=1)

        # Compute the outputs
        outputs = self.linear1(inputs_concat)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        outputs = self.relu(outputs)
        outputs = self.linear3(outputs)
        outputs = self.log_sigmoid(outputs)
        # Cap the reward to -50, enough for machine precision when in exp
        outputs = th.clamp(outputs, min=-50)

        if self.scale:  # Scale [-np.inf, 0] to [-1, 0] for RL training
            outputs = th.tanh(outputs)
        outputs = self.squeeze(outputs)

        assert outputs.shape == state.shape[:1]

        return outputs


class SqueezeLayer(th.nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x) -> th.Tensor:
        """
        Forward pass of the squeeze layer.

        Args:
            x: A tensor to squeeze.

        Returns:
            The squeezed tensor.
        """
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value
