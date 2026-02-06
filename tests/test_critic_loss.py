import jax
import jax.numpy as jnp
from lql.utils.critic_loss import get_tdn_critic_loss, get_lql_critic_loss
from einops import rearrange

def make_synthetic_data(
    batch_size,
    seq_len,
    num_critics,
    action_chunk_size=1,
    action_chunk_eval_interval=1,
):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    rewards = jax.random.uniform(keys[0], (batch_size, seq_len), minval=1.0, maxval=15.0)
    q_a_star_next = jax.random.uniform(keys[1], (batch_size, seq_len), minval=10.0, maxval=50.0)
    completion_mask = jax.random.bernoulli(keys[2], p=0.1, shape=(batch_size, seq_len)).astype(jnp.bool_)
    continuation_mask = jax.random.bernoulli(keys[3], p=0.9, shape=(batch_size, seq_len)).astype(jnp.bool_) & ~completion_mask

    q = jax.random.uniform(keys[4], (num_critics, batch_size, seq_len), minval=10.0, maxval=50.0)

    eval_chunk_start_idx = jnp.arange(0, rewards.shape[1], action_chunk_size * action_chunk_eval_interval)
    q = q[:, :, eval_chunk_start_idx]
    eval_chunk_end_idx = eval_chunk_start_idx + action_chunk_size - 1
    q_a_star_next = q_a_star_next[:, eval_chunk_end_idx]
    return q, q_a_star_next, rewards, completion_mask, continuation_mask

def test_lql_tdn_equivalence_seq_1():
    """
    When seq_len=1, the TDn critic loss should be equivalent to the LQL
    critic loss.
    """
    discount = 0.95
    q, q_a_star_next, rewards, completion_mask, continuation_mask = make_synthetic_data(
        batch_size=64,
        seq_len=1,
        num_critics=2,
    )
    tdn_loss = get_tdn_critic_loss(
        q.squeeze(-1), # expects no seq dim
        q_a_star_next.squeeze(-1), # expects no seq dim
        rewards, 
        (~completion_mask).astype(float), # uses terminals / masks convention
        (~continuation_mask).astype(float), 
        discount
    )[0]
    lql_loss = get_lql_critic_loss(
        q,
        q_a_star_next,
        rewards,
        completion_mask,
        continuation_mask,
        discount
    )[0]

    assert jnp.allclose(tdn_loss, lql_loss, atol=1e-5), f"TDn loss: {tdn_loss}, LQL loss: {lql_loss}"

def test_lql_tdn_equivalence_hinge_weight_0():
    """
    When the lql hinge weight is 0, the loss is equivalent to computing the TD
    loss on the data with the sequence dimension folded into the batch
    dimension.
    """
    discount = 0.95
    q, q_a_star_next, rewards, completion_mask, continuation_mask = make_synthetic_data(
        batch_size=64,
        seq_len=4,
        num_critics=2,
    )

    lql_loss_seq = get_lql_critic_loss(
        q,
        q_a_star_next,
        rewards,
        completion_mask,
        continuation_mask,
        discount,
        hinge_loss_weight=0.0,
    )[0]

    q_flat = rearrange(q, 'num_critics batch seq -> num_critics (batch seq) 1')
    q_a_star_next_flat = rearrange(q_a_star_next, 'batch seq -> (batch seq) 1')
    rewards_flat = rearrange(rewards, 'batch seq -> (batch seq) 1')
    completion_mask_flat = rearrange(completion_mask, 'batch seq -> (batch seq) 1')
    continuation_mask_flat = rearrange(continuation_mask, 'batch seq -> (batch seq) 1')

    # First, compute the 1-step TD loss with the lql loss function 
    lql_loss_flat = get_lql_critic_loss(
        q_flat,
        q_a_star_next_flat,
        rewards_flat,
        completion_mask_flat,
        continuation_mask_flat,
        discount,
    )[0]

    assert jnp.allclose(lql_loss_seq, lql_loss_flat, atol=1e-5), f"With seq dim: {lql_loss_seq}, without: {lql_loss_flat}"

    # Then, compute the same thing with the tdn loss function
    tdn_loss = get_tdn_critic_loss(
        q_flat.squeeze(-1), # expects no seq dim
        q_a_star_next_flat.squeeze(-1), # expects no seq dim
        rewards_flat, 
        (~completion_mask_flat).astype(float), # uses terminals / masks convention
        (~continuation_mask_flat).astype(float), 
        discount
    )[0]

    assert jnp.allclose(tdn_loss, lql_loss_seq, atol=1e-5), f"TDn loss: {tdn_loss}, LQL loss: {lql_loss_seq}"

    
    
def test_lql_tdn_equivalence_action_chunking():
    """
    lql with chunking is equivalent to tdn when n is equal to the chunk size,
    given the same q values at the boundaries of each chunk
    """
    discount = 0.95
    chunk_size = 5
    batch_size = 64
    num_critics = 2

    q, q_a_star_next, rewards, completion_mask, continuation_mask = make_synthetic_data(
        batch_size=batch_size,
        seq_len=chunk_size,
        num_critics=num_critics,
        action_chunk_size=chunk_size,
    )
    assert q.shape == (num_critics, batch_size, 1), q.shape
    assert rewards.shape == (batch_size, chunk_size), rewards.shape

    # Set nonfinal transitions in each sequence to noncompletion and
    # continuation, as TDn normalizes by tensor shape while LQL normalizes by
    # number of valid terms
    completion_mask = completion_mask.at[:, :-1].set(False)
    continuation_mask = continuation_mask.at[:, :-1].set(True)

    tdn_loss = get_tdn_critic_loss(
        q.squeeze(-1), # expects no seq dim
        q_a_star_next.squeeze(-1), # expects no seq dim
        rewards, 
        (~completion_mask).astype(float), # uses terminals / masks convention
        (~continuation_mask).astype(float), 
        discount
    )[0]
    lql_loss = get_lql_critic_loss(
        q,
        q_a_star_next,
        rewards,
        completion_mask,
        continuation_mask,
        discount,
        action_chunk_size=chunk_size,
    )[0]

    assert jnp.allclose(tdn_loss, lql_loss, atol=1e-5), f"TDn loss: {tdn_loss}, LQL loss: {lql_loss}"