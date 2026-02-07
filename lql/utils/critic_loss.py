import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum
from jaxtyping import Array, Float, Int, jaxtyped, Bool
from beartype import beartype
from beartype.typing import Tuple, Dict


@jaxtyped(typechecker=beartype)
def get_utils_to_seq_end(
    rewards: Float[Array, 'batch seq'],
    discount: float,
) -> Float[Array, 'batch seq_plus_one']:
    batch_size, seq_len = rewards.shape

    utils_to_seq_end = jnp.zeros((batch_size, seq_len), dtype=float)

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
        [utils_to_seq_end, jnp.zeros((batch_size, 1), dtype=float)],
        axis=1,
    )
    
    return utils_to_seq_end

@jaxtyped(typechecker=beartype)
def get_chunk_utils(
    rewards: Float[Array, 'batch seq'],
    utils_to_seq_end: Float[Array, 'batch seq_plus_one'],
    completion_mask: Bool[Array, 'batch seq'],
    continuation_mask: Bool[Array, 'batch seq'],
    discount: float,
    action_chunk_size: int,
) -> Tuple[
        Float[Array, 'batch chunk'],
        Bool[Array, 'batch chunk'],
        Bool[Array, 'batch chunk'],
        Bool[Array, 'batch chunk'],
    ]:
    """
    Compute the utilities from each action chunk start (observation at first
    index in chunk) to end (next_observation at last index in chunk).
    """
    batch_size, seq_len = rewards.shape
    assert utils_to_seq_end.shape == (batch_size, seq_len + 1)
    num_chunks = seq_len // action_chunk_size

    chunk_start_idx = jnp.arange(0, seq_len, action_chunk_size)
    # This will use the util to seq end from the next chunk, which may not be
    # part of the same trajectory even if the current chunk is valid, but this
    # is okay because the util is to *sequence end*, not episode or trajectory
    # end.
    chunk_utils = utils_to_seq_end[:, chunk_start_idx] - (
        discount ** action_chunk_size * utils_to_seq_end[:, chunk_start_idx + action_chunk_size]
    )
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
    chunk_completion_mask = completion_mask[:, chunk_start_idx + action_chunk_size - 1]
    chunk_continuation_mask = continuation_mask[:, chunk_start_idx + action_chunk_size - 1]
    return chunk_utils, chunk_valids, chunk_completion_mask, chunk_continuation_mask

@jaxtyped(typechecker=beartype)
def all_between(A: Bool[Array, 'n']) -> Bool[Array, 'n n']:

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


@jaxtyped(typechecker=beartype)
def get_hinge_loss_for_critic(
    q: Float[Array, 'batch eval_chunk'],
    v_next: Float[Array, 'batch eval_chunk'],
    utils_to_seq_end: Float[Array, 'batch seq_plus_one'],
    chunk_utils: Float[Array, 'batch chunk'],
    chunk_valids: Bool[Array, 'batch chunk'],
    chunk_completion_mask: Bool[Array, 'batch chunk'],
    chunk_continuation_mask: Bool[Array, 'batch chunk'],
    discount: float,
    action_chunk_size: int,
    action_chunk_eval_interval: int,
) -> Tuple[Float[Array, ''], Dict]:
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
    lower_bound_errors = jnp.maximum(
        mixed_util_from_eval_chunk_pre - util_from_eval_chunk_pre,
        0.0,
    ) ** 2 * lower_bound_diffs_valid
    lower_bound_denom = jnp.maximum(jnp.sum(lower_bound_diffs_valid.astype(jnp.int32)), 1)
    lower_bound_loss = jnp.sum(lower_bound_errors) / lower_bound_denom

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
    upper_bound_errors = jnp.maximum(
        mixed_util_from_eval_chunk_end_pre - util_from_eval_chunk_end_pre,
        0.0,
    ) ** 2 * upper_bound_diffs_valid
    upper_bound_denom = jnp.maximum(jnp.sum(upper_bound_diffs_valid.astype(jnp.int32)), 1)
    upper_bound_loss = jnp.sum(upper_bound_errors) / upper_bound_denom

    return lower_bound_loss + upper_bound_loss, {
        "lower_bound_loss": lower_bound_loss,
        "upper_bound_loss": upper_bound_loss,
        "num_valid_lower_bound_terms": lower_bound_denom,
        "num_valid_upper_bound_terms": upper_bound_denom,
    }

@jaxtyped(typechecker=beartype)
def get_hinge_loss(
    q: Float[Array, 'critic batch eval_chunk'],
    v_next: Float[Array, 'batch eval_chunk'],
    utils_to_seq_end: Float[Array, 'batch seq_plus_one'],
    chunk_utils: Float[Array, 'batch chunk'],
    chunk_valids: Bool[Array, 'batch chunk'],
    chunk_completion_mask: Bool[Array, 'batch chunk'],
    chunk_continuation_mask: Bool[Array, 'batch chunk'],
    discount: float,
    action_chunk_size: int,
    action_chunk_eval_interval: int,
) -> Tuple[Float[Array, ''], Dict]:
    num_critics, batch_size, num_eval_chunks = q.shape

    # vmap across ensemble dimension
    loss_per_critic, info_per_critic = jax.vmap(
        get_hinge_loss_for_critic,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )(
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
    assert loss_per_critic.shape == (num_critics,)
    assert all(v.shape[0] == num_critics for v in info_per_critic.values())

    return jnp.mean(loss_per_critic), jax.tree_util.tree_map(jnp.mean, info_per_critic)

@jaxtyped(typechecker=beartype)
def get_td_loss(
    q: Float[Array, 'critic batch eval_chunk'],
    v_next: Float[Array, 'batch eval_chunk'],
    chunk_utils: Float[Array, 'batch chunk'],
    chunk_valids: Bool[Array, 'batch chunk'],
    chunk_completion_mask: Bool[Array, 'batch chunk'],
    discount: float,
    action_chunk_size: int,
    action_chunk_eval_interval: int,
) -> Tuple[Float[Array, ''], Dict]:
    num_critics, batch_size, num_eval_chunks = q.shape
    # Select chunks with value evaluations
    eval_chunk_utils, eval_chunk_valids, eval_chunk_completion_mask = jax.tree_util.tree_map(
        lambda x: x[:, ::action_chunk_eval_interval],
        (chunk_utils, chunk_valids, chunk_completion_mask),
    )
    targets = (eval_chunk_utils + v_next * (discount ** action_chunk_size) * (1 - eval_chunk_completion_mask.astype(float)))
    assert targets.shape == eval_chunk_valids.shape == (batch_size, num_eval_chunks)
    td_loss = jnp.sum(
        (
            q - targets
        ) ** 2 * eval_chunk_valids
    ) / jnp.maximum(jnp.sum(eval_chunk_valids.astype(jnp.int32)), 1) / num_critics
    return td_loss, {
        "td_target_mean": jnp.sum(targets * eval_chunk_valids) / jnp.maximum(jnp.sum(eval_chunk_valids), 1),
        "num_valid_td_terms": jnp.sum(eval_chunk_valids.astype(jnp.int32)),
    }

@jaxtyped(typechecker=beartype)
def get_lql_critic_loss(
    q: Float[Array, 'critic batch eval_chunk'],
    v_next: Float[Array, 'batch eval_chunk'],
    rewards: Float[Array, 'batch seq'],
    completion_mask: Bool[Array, 'batch seq'],
    continuation_mask: Bool[Array, 'batch seq'],
    discount: float,
    action_chunk_size: int = 1,
    action_chunk_eval_interval: int = 1,
    hinge_loss_weight: float = 1.0,
) -> Tuple[Float[Array, ''], Dict]:
    """
    Params:
        v_next: Assumed to already be reduced across ensemble dimension.
        action_chunk_size: Number of actions for a chunked policy / value
            function. This is also the number of actions between each q and its
            corresponding v_next.
        action_chunk_eval_interval: Number of chunks between each evaluation of
            the chunked value function.
    """
    batch_size, seq_len = rewards.shape
    assert seq_len % (action_chunk_size * action_chunk_eval_interval) == 0

    # Set completions to discontinuations
    continuation_mask = continuation_mask & ~completion_mask

    utils_to_seq_end = get_utils_to_seq_end(
        rewards,
        discount,
    )

    chunk_utils, chunk_valids, chunk_completion_mask, chunk_continuation_mask = get_chunk_utils(
        rewards,
        utils_to_seq_end,
        completion_mask,
        continuation_mask,
        discount,
        action_chunk_size,
    )

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


@jaxtyped(typechecker=beartype)
def get_tdn_target_q_idx(
    terminals: Float[Array, 'batch seq'],
) -> Int[Array, 'batch']:
    # Find the first terminal transition in each sequence or the last
    # transition if no terminal is present.
    batch_size, seq_len = terminals.shape
    return jnp.where(
        jnp.any(terminals, axis=1),
        jnp.argmax(terminals, axis=1),
        seq_len - 1,
    )

@jaxtyped(typechecker=beartype)
def get_tdn_critic_loss(
    q: Float[Array, 'critic batch'],
    v_next: Float[Array, 'batch'],
    rewards: Float[Array, 'batch seq'],
    masks: Float[Array, 'batch seq'],
    terminals: Float[Array, 'batch seq'],
    discount: float,
) -> Tuple[Float[Array, ''], Dict]:

    """ v_next is assumed to be computed at the first terminal
    next_observation in each sequence, or the last next_observation if no
    terminal is present """

    num_critics = q.shape[0]
    batch_size, seq_len = rewards.shape

    # Construct target for TD-n loss by stepping through sequence and
    # conditionally accumulating

    td_target = jnp.zeros((batch_size,), dtype=float)
    # Whether to stop adding rewards to target
    target_accum_done = jnp.zeros((batch_size,), dtype=float)

    q_target_idx = get_tdn_target_q_idx(terminals)

    for i in range(0, seq_len):
        td_target = td_target + rewards[:, i] * (discount ** i) * (1.0 - target_accum_done)

        # If we (a) reach the q target idx (b) it's not a completion, and (c) we haven't
        # already stopped accumulating: then we should add the bootstrapped
        # value to the target
        td_target = td_target + v_next * (discount ** (i + 1)) * (q_target_idx == i) * masks[:, i] * (1.0 - target_accum_done)

        # If completion or terminal, then we should stop accumulating target
        target_accum_done = jnp.maximum(
            jnp.maximum(target_accum_done, 1.0 - masks[:, i]),
            terminals[:, i],
        )


    q_loss_ens = jnp.square(q - td_target)
    assert q_loss_ens.shape == (num_critics, batch_size)
    q_loss = q_loss_ens.mean()

    info = {
        'td_loss/td_target_mean': td_target.mean(),
    }

    return q_loss, info
    