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
    """Replay buffer with JIT-compiled or numpy sampling and insertion.

    When jit_data=True, data is stored as JAX arrays on the default device
    and operations are JIT-compiled. When jit_data=False, data is stored as
    numpy arrays and operations use numpy indexing.
    """
    data: dict
    pointer: jax.Array
    size: jax.Array
    max_size: int = flax.struct.field(pytree_node=False)
    prev_was_terminal: jax.Array
    jit_data: bool = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, transition, max_size, jit_data=False):
        """Create an empty buffer.

        Args:
            transition: Example transition dict (numpy or JAX arrays, unbatched).
            max_size: Buffer capacity.
            jit_data: If True, store as JAX arrays on default device.
        """
        def make_buf(example):
            example = np.asarray(example)
            buf = np.zeros((max_size, *example.shape), dtype=example.dtype)
            if jit_data:
                return jnp.asarray(buf)
            return buf

        data = jax.tree.map(make_buf, transition)
        if jit_data:
            pointer = jnp.int32(0)
            size = jnp.int32(0)
            prev_was_terminal = jnp.bool_(True)
        else:
            pointer = 0
            size = 0
            prev_was_terminal = True
        return cls(
            data=data, pointer=pointer, size=size,
            max_size=max_size, prev_was_terminal=prev_was_terminal,
            jit_data=jit_data,
        )

    @classmethod
    def create_from_initial_dataset(cls, dataset_dict, max_size, jit_data=False):
        """Create a buffer pre-filled with an existing dataset.

        Args:
            dataset_dict: Dict of numpy arrays, each with first dim = num transitions.
            max_size: Buffer capacity.
            jit_data: If True, store as JAX arrays on default device.
        """
        init_size = len(next(iter(dataset_dict.values())))
        fill_size = min(init_size, max_size)
        print(f"Initializing buffer with {fill_size} transitions (max_size={max_size})")

        def make_buf(init_arr):
            buf = np.zeros((max_size, *init_arr.shape[1:]), dtype=init_arr.dtype)
            buf[:fill_size] = init_arr[:fill_size]
            if jit_data:
                return jnp.asarray(buf)
            return buf

        data = jax.tree.map(make_buf, dataset_dict)
        if jit_data:
            pointer = jnp.int32(fill_size % max_size)
            size = jnp.int32(fill_size)
            prev_was_terminal = jnp.bool_(True)
        else:
            pointer = fill_size % max_size
            size = fill_size
            prev_was_terminal = True
        return cls(
            data=data, pointer=pointer, size=size,
            max_size=max_size, prev_was_terminal=prev_was_terminal,
            jit_data=jit_data,
        )

    def add_transition(self, transition):
        """Add one transition. Returns updated buffer."""
        if self.jit_data:
            return self._add_transition_jit(transition)
        return self._add_transition_numpy(transition)

    @partial(jax.jit, donate_argnums=(0,))
    def _add_transition_jit(self, transition):
        this_true_terminal = transition['terminals']

        transition = {**transition, 'terminals': jnp.float32(1.0)}

        new_data = jax.tree.map(
            lambda buf, val: buf.at[self.pointer].set(val),
            self.data, transition,
        )

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

    def _add_transition_numpy(self, transition):
        pointer = int(self.pointer)
        this_true_terminal = float(transition['terminals'])

        for k in self.data:
            if k == 'terminals':
                self.data[k][pointer] = np.float32(1.0)
            else:
                self.data[k][pointer] = transition[k]

        if not self.prev_was_terminal:
            prev_pointer = (pointer - 1) % self.max_size
            self.data['terminals'][prev_pointer] = np.float32(0.0)

        new_pointer = (pointer + 1) % self.max_size
        new_size = min(int(self.size) + 1, self.max_size)

        return self.replace(
            pointer=new_pointer,
            size=new_size,
            prev_was_terminal=bool(this_true_terminal),
        )

    def sample_contiguous(self, key, batch_size, sequence_length):
        """Sample contiguous sequences. Dispatches to numpy or JIT path."""
        if self.jit_data:
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
