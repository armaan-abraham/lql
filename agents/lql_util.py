import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum


def get_utils_to_seq_end(
    rewards: jnp.ndarray,
    discount: float,
):
    batch_size, seq_len = rewards.shape

    utils_to_seq_end = jnp.zeros((batch_size, seq_len), dtype=rewards.dtype)

    # These utils are only meaningful when used to compute relative utils
    # between transitions in the same trajectory
    for t in reversed(range(seq_len)):
        utils_to_seq_end = utils_to_seq_end.at[:, t].set(
            rewards[:, t] + discount * jnp.where(
                t + 1 < seq_len,
                utils_to_seq_end[:, t + 1],
                0.0,
            )
        )
    
    # Pad with an extra zero at the end for easier indexing
    utils_to_seq_end = jnp.concatenate(
        [utils_to_seq_end, jnp.zeros((batch_size, 1), dtype=rewards.dtype)],
        axis=1,
    )
    
    return utils_to_seq_end

def get_chunk_utils(
    rewards: jnp.ndarray,
    utils_to_seq_end: jnp.ndarray,
    completion_mask: jnp.ndarray,
    continuation_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int,
):
    """
    Compute the utilities in each action chunk.
    """
    batch_size, seq_len = rewards.shape
    num_chunks = seq_len // action_chunk_size
    assert utils_to_seq_end.shape == (batch_size, seq_len + 1)

    chunk_start_idx = jnp.arange(0, seq_len, action_chunk_size)
    # This will use the util to seq end from the next chunk, which may not be
    # part of the same trajectory even if the current chunk is valid, but this
    # is okay because the util is to *sequence end*, not episode or trajectory
    # end.
    chunk_utils = utils_to_seq_end[:, chunk_start_idx] - (
        discount ** action_chunk_size * utils_to_seq_end[:, chunk_start_idx + action_chunk_size]
    )
    assert chunk_utils.shape == (batch_size, num_chunks)
    continuation_mask_by_chunk = rearrange(
        continuation_mask,
        "batch (num_chunks chunk_size) -> batch num_chunks chunk_size",
        chunk_size=action_chunk_size,
    )
    # A chunk is valid if all non-final transitions are continuations
    chunk_valids = jnp.all(
        continuation_mask_by_chunk[:, :, :-1],
        axis=-1,
    )
    assert chunk_valids.shape == (batch_size, num_chunks)
    chunk_completion_mask = completion_mask[:, chunk_start_idx + action_chunk_size - 1]
    assert chunk_completion_mask.shape == (batch_size, num_chunks)
    chunk_continuation_mask = continuation_mask[:, chunk_start_idx + action_chunk_size - 1]
    return chunk_utils, chunk_valids, chunk_completion_mask, chunk_continuation_mask

def all_between(A):
    assert A.dtype == jnp.bool_

    n = A.shape[0]
    
    cumsum = jnp.cumsum(A)
    
    # Create indices
    i = jnp.arange(n)[:, None]  # (n, 1)
    j = jnp.arange(n)[None, :]  # (1, n)
    
    # Prepend 0 to cumsum for easier indexing
    cumsum_padded = jnp.concatenate([jnp.array([0]), cumsum])
    
    lo = jnp.minimum(i, j) + 1
    hi = jnp.maximum(i, j)
    
    range_sum = jnp.maximum(cumsum_padded[hi] - cumsum_padded[lo], 0)
    range_len = jnp.maximum(hi - lo, 0)
    
    B = range_sum == range_len
    
    return B

def get_hinge_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    utils_to_seq_end: jnp.ndarray,
    chunk_utils: jnp.ndarray,
    chunk_valids: jnp.ndarray,
    chunk_completion_mask: jnp.ndarray,
    chunk_continuation_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int,
    action_chunk_eval_interval: int,
):
    """
    q and v_next are expected to be provided for each eval chunk (q at chunk
    start and v_next at chunk end).
    """
    seq_len = utils_to_seq_end.shape[1] - 1 # -1 for padding
    batch_size, num_eval_chunks = q.shape

    # Select chunks with value evaluations
    eval_chunk_utils, eval_chunk_valids, eval_chunk_completion_mask, eval_chunk_continuation_mask = jax.tree_util.tree_map(
        lambda x: x[:, ::action_chunk_eval_interval],
        (chunk_utils, chunk_valids, chunk_completion_mask, chunk_continuation_mask),
    )

    eval_chunk_start_idx = jnp.arange(0, seq_len, (action_chunk_size * action_chunk_eval_interval))
    eval_chunk_start_utils_to_seq_end = utils_to_seq_end[:, eval_chunk_start_idx]
    pairwise_time_diffs = (
         repeat(
            jnp.arange(num_eval_chunks),
            "chunk_post -> chunk_pre chunk_post",
            chunk_pre=num_eval_chunks,
        ) - repeat(
            jnp.arange(num_eval_chunks),
            "chunk_pre -> chunk_pre chunk_post",
            chunk_post=num_eval_chunks,
        )
    ) * (action_chunk_size * action_chunk_eval_interval)
    occurs_after = pairwise_time_diffs > 0
    # Compute pairwise utilities between each eval chunk starting point. Element
    # (i,j) for i < j is the utility from chunk i to j.
    eval_chunk_start_pairwise_utils = repeat(
        eval_chunk_start_utils_to_seq_end,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_eval_chunks,
    ) - repeat(
        eval_chunk_start_utils_to_seq_end,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_eval_chunks,
    ) * discount ** pairwise_time_diffs

    # Compute mask for if all chunks between each pair of eval chunks is a valid
    # nonterminal chunk. The boundary chunks will be handled for lower bound and
    # upper bound losses separately.
    intermediates_valid = jax.vmap(all_between, in_axes=0)(
        chunk_valids & chunk_continuation_mask
    )[:, ::action_chunk_eval_interval, ::action_chunk_eval_interval]
    occurs_after = pairwise_time_diffs > 0
    assert intermediates_valid.shape == (batch_size, num_eval_chunks, num_eval_chunks)
    base_diffs_valid = occurs_after & intermediates_valid

    # Compute lower bound loss by comparing earlier q values to later v_next
    # values.

    # Compute the estimated utility-to-go (observed->optimal policy) from the observed
    # utils between chunks plus the v_next value function estimate at the later
    # chunk.
    eval_chunk_post_util_to_go = repeat(
        eval_chunk_utils + v_next * (discount ** action_chunk_size) * (1 - eval_chunk_completion_mask.astype(q.dtype)),
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_eval_chunks,
    )
    eval_chunk_post_util_to_go_discount = discount ** pairwise_time_diffs
    mixed_util_from_eval_chunk_pre = eval_chunk_start_pairwise_utils + eval_chunk_post_util_to_go * eval_chunk_post_util_to_go_discount
    util_from_eval_chunk_pre = repeat(
        q,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_eval_chunks,
    )
    # All intermediate chunks must be valid nonterminals, the start chunk must
    # be a valid nonterminal, and the end chunk must be valid (possibly
    # terminal).
    lower_bound_diffs_valid = base_diffs_valid & repeat(
        eval_chunk_valids & eval_chunk_continuation_mask,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_eval_chunks,
    ) & repeat(
        eval_chunk_valids,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_eval_chunks,
    )
    lower_bound_loss = jnp.sum(
        jnp.maximum(
            mixed_util_from_eval_chunk_pre - util_from_eval_chunk_pre,
            0.0,
        ) ** 2 * lower_bound_diffs_valid
    ) / jnp.maximum(jnp.sum(lower_bound_diffs_valid.astype(jnp.int32)), 1)

    # Compute upper bound loss by comparing later q values to earlier v_next values.

    # Compute estimated utility-to-go from the end of the earlier chunk
    util_from_eval_chunk_end_pre = repeat(
        v_next,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_eval_chunks,
    )

    # Compute the estimated utility-to-go from the end of the earlier chunk
    # using the observed utils between chunks plus the q value function estimate
    # at the later chunk.
    # We need to subtract the util of the earlier chunk from the relative util
    # due to using v_next, which is at the end of the earlier chunk.
    mixed_util_from_eval_chunk_end_pre = (eval_chunk_start_pairwise_utils - repeat(
        eval_chunk_utils,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_eval_chunks,
    )) / discount ** action_chunk_size + repeat(
        q,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_eval_chunks,
    ) * discount ** (pairwise_time_diffs - action_chunk_size)
    # All intermediate chunks must be valid nonterminals, the start chunk must
    # be nonterminal (possibly invalid), and the end chunk must be valid
    # (possibly terminal).
    upper_bound_diffs_valid = base_diffs_valid & repeat(
        eval_chunk_continuation_mask,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_eval_chunks,
    ) & repeat(
        eval_chunk_valids,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_eval_chunks,
    )
    upper_bound_loss = jnp.sum(
        jnp.maximum(
            mixed_util_from_eval_chunk_end_pre - util_from_eval_chunk_end_pre,
            0.0,
        ) ** 2 * upper_bound_diffs_valid
    ) / jnp.maximum(jnp.sum(upper_bound_diffs_valid.astype(jnp.int32)), 1)
    return lower_bound_loss + upper_bound_loss, {
        "lower_bound_loss": lower_bound_loss,
        "upper_bound_loss": upper_bound_loss,
        "num_valid_lower_bound_terms": jnp.sum(lower_bound_diffs_valid.astype(jnp.int32)),
        "num_valid_upper_bound_terms": jnp.sum(upper_bound_diffs_valid.astype(jnp.int32)),
    }

def get_td_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    chunk_utils: jnp.ndarray,
    chunk_valids: jnp.ndarray,
    chunk_completion_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int,
    action_chunk_eval_interval: int,
):
    # Select chunks with value evaluations
    eval_chunk_utils, eval_chunk_valids, eval_chunk_completion_mask = jax.tree_util.tree_map(
        lambda x: x[:, ::action_chunk_eval_interval],
        (chunk_utils, chunk_valids, chunk_completion_mask),
    )
    targets = (eval_chunk_utils + v_next * (discount ** action_chunk_size) * (1 - eval_chunk_completion_mask.astype(q.dtype)))
    td_loss = jnp.sum(
        (
            q - targets
        ) ** 2 * eval_chunk_valids
    ) / jnp.maximum(jnp.sum(eval_chunk_valids.astype(jnp.int32)), 1)
    return td_loss, {
        "td_target_mean": jnp.sum(targets * eval_chunk_valids) / jnp.maximum(jnp.sum(eval_chunk_valids), 1),
        "num_valid_td_terms": jnp.sum(eval_chunk_valids.astype(jnp.int32)),
    }

def get_lql_critic_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    rewards: jnp.ndarray,
    completion_mask: jnp.ndarray,
    continuation_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int = 1,
    action_chunk_eval_interval: int = 1,
    hinge_loss_weight: float = 1.0,
):
    """
    Params:
        action_chunk_size: Number of actions for a chunked policy / value
        function. This is also the number of actions between each q and its
        corresponding v_next.
        action_chunk_eval_interval: Number of chunks between each evaluation of
        the chunked value function.
    """
    assert jnp.issubdtype(q.dtype, jnp.floating)
    assert jnp.issubdtype(v_next.dtype, jnp.floating)
    assert jnp.issubdtype(rewards.dtype, jnp.floating)
    batch_size, seq_len = rewards.shape
    assert seq_len % (action_chunk_size * action_chunk_eval_interval) == 0
    num_chunks = seq_len // action_chunk_size
    num_eval_chunks = num_chunks // action_chunk_eval_interval
    assert q.shape == v_next.shape == (batch_size, num_eval_chunks)
    assert completion_mask.shape == continuation_mask.shape == (batch_size, seq_len)
    assert completion_mask.dtype == jnp.bool_
    assert continuation_mask.dtype == jnp.bool_

    # Set completions to discontinuations
    continuation_mask = continuation_mask & ~completion_mask

    utils_to_seq_end = get_utils_to_seq_end(
        rewards,
        discount,
    )
    assert utils_to_seq_end.dtype == jnp.float32

    chunk_utils, chunk_valids, chunk_completion_mask, chunk_continuation_mask = get_chunk_utils(
        rewards,
        utils_to_seq_end,
        completion_mask,
        continuation_mask,
        discount,
        action_chunk_size,
    )
    assert chunk_utils.dtype == jnp.float32
    assert chunk_valids.dtype == jnp.bool
    assert chunk_completion_mask.dtype == jnp.bool

    td_loss, td_info = get_td_loss(
        q,
        v_next,
        chunk_utils,
        chunk_valids,
        chunk_completion_mask,
        discount,
        action_chunk_size,
        action_chunk_eval_interval,
    )

    hinge_loss, hinge_info = get_hinge_loss(
        q,
        v_next,
        utils_to_seq_end,
        chunk_utils,
        chunk_valids,
        chunk_completion_mask,
        chunk_continuation_mask,
        discount,
        action_chunk_size,
        action_chunk_eval_interval,
    )

    info = {
        "td_loss": td_loss,
        "hinge_loss": hinge_loss,
    }

    for k, v in td_info.items():
        info[f"td_loss/{k}"] = v
    
    for k, v in hinge_info.items():
        info[f"hinge_loss/{k}"] = v

    return td_loss + hinge_loss * hinge_loss_weight, info

if __name__ == "__main__":
    # Tests
    print("*************************")
    A = jnp.array([1, 1, 1, 1, 0, 1, ]).astype(bool)
    print("A\n", A.astype(jnp.int32))
    print("all between \n", all_between(A).astype(jnp.int32))

    B = jnp.arange(8).reshape(4,2)
    print("B\n", B)
    

    print(rearrange(
        B,
        "(num_chunks chunk_size) dim -> num_chunks (chunk_size dim)",
        chunk_size=2,
    ))

    # # Chunk case
    discount = 0.9
    # rewards = jnp.array(
    #     [
    #         [1, 2, 3, 4, 5, 6, 7, 8],
    #         [5, 7, 9, 11, 13, 15, 17, 19],
    #     ], dtype=jnp.float32
    # )
    # completion_mask = jnp.array(
    #     [
    #         [0, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1],
    #     ]
    # ).astype(bool)
    # continuation_mask = jnp.array(
    #     [
    #         [1, 0, 1, 1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1, 0, 1, 0],
    #     ]
    # ).astype(bool)
    # q = jnp.array(
    #     [
    #         [11, 13, 15, 17, 19, 21, 23, 25],
    #         [42, 44, 46, 48, 50, 52, 54, 56],
    #     ], dtype=jnp.float32
    # )
    # q_a_star_next = jnp.array(
    #     [
    #         [10, 11, 12, 13, 14, 15, 16, 17],
    #         [40, 42, 44, 46, 48, 50, 52, 54],
    #     ], dtype=jnp.float32
    # )
    # action_chunk_size = 2
    # action_chunk_eval_interval = 1
    # eval_chunk_start_idx = jnp.arange(0, rewards.shape[1], action_chunk_size * action_chunk_eval_interval)
    # q = q[:, eval_chunk_start_idx]
    # eval_chunk_end_idx = eval_chunk_start_idx + action_chunk_size - 1
    # print("q")
    # print(q)
    # q_a_star_next = q_a_star_next[:, eval_chunk_end_idx]
    # print("q_a_star_next")
    # print(q_a_star_next)
    # loss = get_lql_loss(
    #     q,
    #     q_a_star_next,
    #     rewards,
    #     completion_mask,
    #     continuation_mask,
    #     discount,
    #     action_chunk_size=action_chunk_size,
    #     action_chunk_eval_interval=action_chunk_eval_interval,
    # )
    # print("LQL loss", loss)


    # # single action case
    # discount = 0.9 ** 2
    # rewards = jnp.array(
    #     [
    #         [2.7999992, 6.5999985, 10.399999, 14.20],
    #         [11.299995, 18.899998, 26.5, 34.1],
    #     ], dtype=jnp.float32
    # )
    # completion_mask = jnp.array(
    #     [
    #         [1, 0, 0, 0],
    #         [0, 0, 0, 1],
    #     ]
    # ).astype(bool)
    # continuation_mask = jnp.array(
    #     [
    #         [0, 1, 1, 0],
    #         [1, 1, 0, 0],
    #     ]
    # ).astype(bool)
    # q = jnp.array(
    #     [
    #         [11, 15, 19, 23],
    #         [42, 46, 50, 54],
    #     ], dtype=jnp.float32
    # )
    # q_a_star_next = jnp.array(
    #     [
    #         [11, 13, 15, 17],
    #         [42, 46, 50, 54],
    #     ], dtype=jnp.float32
    # )

    # Shape constants
    BATCH_SIZE = 2
    SEQ_LEN = 4

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    rewards = jax.random.uniform(keys[0], (BATCH_SIZE, SEQ_LEN), minval=1.0, maxval=15.0)
    q_a_star_next = jax.random.uniform(keys[1], (BATCH_SIZE, SEQ_LEN), minval=10.0, maxval=50.0)

    completion_mask = jnp.zeros_like(rewards).astype(jnp.bool_)
    completion_mask = completion_mask.at[jnp.arange(BATCH_SIZE), -1].set(jax.random.bernoulli(keys[2], p=0.5, shape=(BATCH_SIZE,)).astype(jnp.bool_))
    print("completion mask\n", completion_mask.astype(jnp.int32))
    continuation_mask = jnp.ones_like(completion_mask).astype(jnp.bool_)

    q = jax.random.uniform(keys[4], (BATCH_SIZE, SEQ_LEN), minval=10.0, maxval=50.0)

    action_chunk_size = 1
    action_chunk_eval_interval = 2
    eval_chunk_start_idx = jnp.arange(0, rewards.shape[1], action_chunk_size * action_chunk_eval_interval)
    q = q[:, eval_chunk_start_idx]
    eval_chunk_end_idx = eval_chunk_start_idx + action_chunk_size - 1
    print("q")
    print(q)
    q_a_star_next = q_a_star_next[:, eval_chunk_end_idx]
    print("q_a_star_next")
    print(q_a_star_next)

    loss = get_lql_critic_loss(
        q,
        q_a_star_next,
        rewards,
        completion_mask,
        continuation_mask,
        discount,
        action_chunk_size=action_chunk_size,
        action_chunk_eval_interval=action_chunk_eval_interval,
    )
    print("LQL loss", loss)