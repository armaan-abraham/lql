from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def sample_contiguous(self, batch_size, sequence_length):
        """Sample a batch of sequences, possibly crossing episode boundaries."""

        # Sample sequence starting locations
        idxs = np.random.randint(self.size - sequence_length, size=batch_size)

        # Compute remaining indices for all sequences
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten() 

        # Batch fetch data to avoid loops
        batch_observations = self['observations'][all_idxs].reshape(batch_size, sequence_length, *self['observations'].shape[1:])
        batch_next_observations = self['next_observations'][all_idxs].reshape(batch_size, sequence_length, *self['next_observations'].shape[1:])
        batch_actions = self['actions'][all_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        batch_rewards = self['rewards'][all_idxs].reshape(batch_size, sequence_length, *self['rewards'].shape[1:])
        batch_masks = self['masks'][all_idxs].reshape(batch_size, sequence_length, *self['masks'].shape[1:])
        batch_terminals = self['terminals'][all_idxs].reshape(batch_size, sequence_length, *self['terminals'].shape[1:])

        # Assert next observations are shifted by one timestep for each sequence
        assert np.all(np.all((batch_next_observations[:, :-1] == batch_observations[:, 1:]), axis=-1) | (batch_terminals[:, :-1] == 1) | (batch_masks[:, :-1] == 0))

        return dict(
            observations=batch_observations,
            next_observations=batch_next_observations,
            actions=batch_actions,
            masks=batch_masks,
            rewards=batch_rewards,
            terminals=batch_terminals,
        )

class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0
        self.last_transition_true_terminal = True

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        # When adding transitions online, we need to keep the boundary between
        # the current episode and the episode after the pointer properly
        # terminal-separated.
        this_transition_true_terminal = transition['terminals']

        transition['terminals'] = 1.0

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)

        # If last transition was not a true terminal, we need to set the
        # terminal flag to 0 for it.
        if not self.last_transition_true_terminal:
            prev_pointer = (self.pointer - 1) % self.max_size
            self._dict['terminals'][prev_pointer] = 0.0

        self.last_transition_true_terminal = this_transition_true_terminal
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
