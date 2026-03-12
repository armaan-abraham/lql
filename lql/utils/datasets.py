from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Lightweight dataset container for data loading."""

    @classmethod
    def create(cls, **fields):
        data = fields
        assert 'observations' in data
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)


class ReplayBuffer(flax.struct.PyTreeNode):
    """Replay buffer stored as JAX arrays.

    Sampling and insertion are JIT-compiled. The buffer is a valid pytree
    and can be passed into/out of JIT-compiled functions.
    """
    data: dict
    pointer: jax.Array
    size: jax.Array
    max_size: int = flax.struct.field(pytree_node=False)
    prev_was_terminal: jax.Array

    on_gpu: bool = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, transition, max_size, device=None):
        """Create an empty buffer on the given device.

        Args:
            transition: Example transition dict (numpy or JAX arrays, unbatched).
            max_size: Buffer capacity.
            device: Target device.
        """
        def make_buf(example):
            example = np.asarray(example)
            buf = np.zeros((max_size, *example.shape), dtype=example.dtype)
            return jax.device_put(jnp.asarray(buf), device)

        data = jax.tree.map(make_buf, transition)
        pointer = jax.device_put(jnp.int32(0), device)
        size = jax.device_put(jnp.int32(0), device)
        prev_was_terminal = jax.device_put(jnp.bool_(True), device)
        on_gpu = device is not None and device.platform != 'cpu'
        return cls(
            data=data, pointer=pointer, size=size,
            max_size=max_size, prev_was_terminal=prev_was_terminal,
            on_gpu=on_gpu,
        )

    @classmethod
    def create_from_initial_dataset(cls, dataset_dict, max_size, device=None):
        """Create a buffer pre-filled with an existing dataset.

        Args:
            dataset_dict: Dict of numpy arrays, each with first dim = num transitions.
            max_size: Buffer capacity (must be >= initial dataset size).
            device: Target device.
        """
        init_size = len(next(iter(dataset_dict.values())))

        fill_size = min(init_size, max_size)

        def make_buf(init_arr):
            buf = np.zeros((max_size, *init_arr.shape[1:]), dtype=init_arr.dtype)
            buf[:fill_size] = init_arr[:fill_size]
            return jax.device_put(jnp.asarray(buf), device)

        data = jax.tree.map(make_buf, dataset_dict)
        pointer = jax.device_put(jnp.int32(fill_size % max_size), device)
        size = jax.device_put(jnp.int32(fill_size), device)
        prev_was_terminal = jax.device_put(jnp.bool_(True), device)
        on_gpu = device is not None and device.platform != 'cpu'
        return cls(
            data=data, pointer=pointer, size=size,
            max_size=max_size, prev_was_terminal=prev_was_terminal,
            on_gpu=on_gpu,
        )

    @partial(jax.jit, donate_argnums=(0,))
    def add_transition(self, transition):
        """Add one transition. Returns updated buffer."""
        this_true_terminal = transition['terminals']

        # Mark current position as terminal
        transition = {**transition, 'terminals': jnp.float32(1.0)}

        # Write transition at pointer
        new_data = jax.tree.map(
            lambda buf, val: buf.at[self.pointer].set(val),
            self.data, transition,
        )

        # If previous transition was NOT a true terminal, clear its terminal flag
        prev_pointer = (self.pointer - 1) % self.max_size
        prev_terminal_val = jnp.where(self.prev_was_terminal, 1.0, 0.0)
        new_data['terminals'] = new_data['terminals'].at[prev_pointer].set(prev_terminal_val)

        new_pointer = (self.pointer + 1) % self.max_size
        new_size = jnp.minimum(self.size + 1, self.max_size)

        return self.replace(
            data=new_data,
            pointer=new_pointer,
            size=new_size,
            prev_was_terminal=jnp.bool_(this_true_terminal),
        )

    def sample_contiguous(self, key, batch_size, sequence_length):
        """Sample contiguous sequences. Dispatches to numpy (CPU) or JIT (GPU)."""
        if self.on_gpu:
            return self._sample_jit(key, batch_size, sequence_length)
        return self._sample_numpy(batch_size, sequence_length)

    @partial(jax.jit, static_argnums=(2, 3))
    def _sample_jit(self, key, batch_size, sequence_length):
        idxs = jax.random.randint(key, (batch_size,), 0, self.size - sequence_length)
        offsets = jnp.arange(sequence_length)
        all_idxs = (idxs[:, None] + offsets[None, :]).flatten()

        def fetch_and_reshape(arr):
            return arr[all_idxs].reshape(batch_size, sequence_length, *arr.shape[1:])

        return jax.tree.map(fetch_and_reshape, self.data)

    def _sample_numpy(self, batch_size, sequence_length):
        size = int(self.size)
        idxs = np.random.randint(size - sequence_length, size=batch_size)
        all_idxs = (idxs[:, None] + np.arange(sequence_length)[None, :]).flatten()

        def fetch_and_reshape(arr):
            return np.asarray(arr)[all_idxs].reshape(batch_size, sequence_length, *arr.shape[1:])

        return jax.tree.map(fetch_and_reshape, self.data)
