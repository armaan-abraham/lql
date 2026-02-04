import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from einops import repeat, rearrange, reduce

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, MLP
from utils.rlpd_utils import TanhNormal, Temperature
from agents.lql_util import get_lql_critic_loss

class LQLAgent(flax.struct.PyTreeNode):
    """Long-horizon Q-learning (LQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        # Batch data must have proper sequence structure
        assert batch['observations'].ndim == 3 # [batch, seq_len, obs_dim]
        assert batch['actions'].ndim == 3 # [batch, seq_len, act_dim]
        assert batch['rewards'].ndim == 2 # [batch, seq_len]
        assert batch['masks'].ndim == 2 # [batch, seq_len]
        assert batch['terminals'].ndim == 2 # [batch, seq_len]
        assert batch['observations'].shape[0:2] == batch['actions'].shape[0:2] == batch['rewards'].shape[0:2] == batch['masks'].shape[0:2] == batch['terminals'].shape[0:2]
        batch_size, seq_len = batch['observations'].shape[0:2]

        batch_select = self.select_obs_and_act_chunks(
            batch,
            self.config['action_chunk_size'],
            self.config['action_chunk_eval_interval'],
        )

        rng, sample_rng = jax.random.split(rng)
        a_star_next = self.sample_actions(batch_select['next_observations'], rng=sample_rng)
        assert a_star_next.shape == (batch_size, self.config['num_eval_chunks_per_seq'], self.config['action_chunk_dim'])
        q_a_star_next_ens = jax.lax.stop_gradient(self.network.select('target_critic')(batch_select['next_observations'], actions=a_star_next))
        assert q_a_star_next_ens.shape == (self.config['num_critics'], batch_size, self.config['num_eval_chunks_per_seq'])
        q_a_star_next = reduce(q_a_star_next_ens, 'ensemble batch chunk -> batch chunk', 'mean')

        q_ens = self.network.select('critic')(batch_select['observations'], actions=batch_select['action_chunks'], params=grad_params)
        assert q_ens.shape == (self.config['num_critics'], batch_size, self.config['num_eval_chunks_per_seq'])

        q_loss, q_loss_info = get_lql_critic_loss(
            q_ens,
            q_a_star_next,
            batch['rewards'],
            ~batch['masks'].astype(bool),
            ~batch['terminals'].astype(bool),
            self.config['discount'],
            self.config['action_chunk_size'],
            self.config['action_chunk_eval_interval'],
            self.config['hinge_loss_weight'],
        )

        return q_loss, {
            'critic_loss': q_loss,
            'q_mean': q_ens.mean(),
            'q_std': q_ens.std(),
            'q_max': q_ens.max(),
            'q_min': q_ens.min(),
            'q_a_star_next_mean': q_a_star_next.mean(),
            'q_a_star_next_std': q_a_star_next.std(),
            'q_a_star_next_max': q_a_star_next.max(),
            'q_a_star_next_min': q_a_star_next.min(),
            **q_loss_info,
        }
    
    @classmethod
    def select_obs_and_act_chunks(cls, batch, action_chunk_size, action_chunk_eval_interval):
        assert batch['observations'].ndim == batch['actions'].ndim == 3
        batch_size, seq_len = batch['observations'].shape[0:2]

        result = {}

        # Observations should come from the first transition in each eval action
        # chunk.
        eval_chunk_start_idx = jnp.arange(0, seq_len, action_chunk_size * action_chunk_eval_interval)
        result['observations'] = batch['observations'][:, eval_chunk_start_idx, :]

        # Actions should be grouped into chunks and then selected at the eval
        # interval.
        action_chunks = rearrange(
            batch['actions'],
            'batch (chunk act) act_dim -> batch chunk (act act_dim)',
            act=action_chunk_size,
        )[:, ::action_chunk_eval_interval, :]
        assert result['observations'].shape[0:2] == action_chunks.shape[0:2], (result['observations'].shape, action_chunks.shape)
        result['action_chunks'] = action_chunks

        # Add valid mask for action chunks
        if 'terminals' in batch:
            if action_chunk_size == 1:
                action_chunk_valids = jnp.ones(action_chunks.shape[:-1], dtype=jnp.bool_)
            else:
                # Action chunks are valid if every nonfinal transition is nonterminal
                action_chunk_terminals = rearrange(
                    batch['terminals'],
                    "batch (chunk act) -> batch chunk act",
                    act=action_chunk_size,
                )[:, ::action_chunk_eval_interval, :].astype(jnp.bool_)
                action_chunk_valids = jnp.all(
                    ~action_chunk_terminals[:, :, :-1],
                    axis=-1,
                )
            result['action_chunk_valids'] = action_chunk_valids

        if 'next_observations' in batch:
            # Next observations should come from the last transition in each eval action chunk.
            eval_chunk_end_idx = eval_chunk_start_idx + (action_chunk_size - 1)
            result['next_observations'] = batch['next_observations'][:, eval_chunk_end_idx, :]

        return result        
    

    def actor_loss(self, batch, grad_params, rng):
        # Batch data must have proper sequence structure
        assert batch['observations'].ndim == 3 # [batch, seq_len, obs_dim]
        assert batch['actions'].ndim == 3 # [batch, seq_len, act_dim]
        assert batch['observations'].shape[0:2] == batch['actions'].shape[0:2]
        batch_size = batch['actions'].shape[0]

        batch_select = self.select_obs_and_act_chunks(
            batch,
            self.config['action_chunk_size'],
            self.config['action_chunk_eval_interval'],
        )
        assert batch_select['observations'].shape == (batch_size, self.config['num_eval_chunks_per_seq'], self.config['obs_dim'])

        if self.config['actor_type'] in ('best-of-n', 'fql'):
            rng, x_rng, t_rng = jax.random.split(rng, 3)

            # BC flow loss.
            x_0 = jax.random.normal(x_rng, (batch_size, self.config['num_eval_chunks_per_seq'], self.config['action_chunk_dim']))
            x_1 = batch_select['action_chunks']
            t = jax.random.uniform(t_rng, (batch_size, self.config['num_eval_chunks_per_seq'], 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = self.network.select('actor')(batch_select['observations'], x_t, t, params=grad_params)

            assert pred.shape[:-1] == batch_select['action_chunk_valids'].shape, (pred.shape, batch_select['action_chunk_valids'].shape)
        
            bc_flow_loss = jnp.mean(
                # Only apply loss to valid action chunks
                (
                    (pred - vel) * batch_select['action_chunk_valids'][..., None]
                ) ** 2
            )
    
            if self.config["actor_type"] == "fql":
                # Distillation loss. No need to filter by valid chunks because
                # the chunks are generated by the policy given only the first
                # observation from the chunk.
                rng, noise_rng = jax.random.split(rng)
                noises = jax.random.normal(noise_rng, (batch_size, self.config['num_eval_chunks_per_seq'], self.config['action_chunk_dim']))
                target_flow_actions = self.compute_flow_actions(batch_select['observations'], noises=noises)
                actor_actions = self.network.select('actor_onestep_flow')(batch_select['observations'], noises, params=grad_params)
                assert actor_actions.shape == (batch_size, self.config['num_eval_chunks_per_seq'], self.config['action_chunk_dim'])
                distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
                
                # Q loss.
                actor_actions = jnp.clip(actor_actions, -1, 1)

                qs = self.network.select(f'critic')(batch_select['observations'], actions=actor_actions)
                q = jnp.mean(qs, axis=0)
                q_loss = -q.mean()
            else:
                distill_loss = jnp.zeros(())
                q_loss = jnp.zeros(())
        
            # Total loss.
            actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_flow_loss': bc_flow_loss,
                'distill_loss': distill_loss,
            }

        else: # gaussian
            actor_dists = self.network.select('actor')(batch_select['observations'], params=grad_params)
            actor_actions = actor_dists.sample(seed=rng)
            log_probs = actor_dists.log_prob(actor_actions)

            # Behavorial cloning loss
            log_probs_mean = actor_dists.log_prob(jnp.clip(batch_select['action_chunks'], -1 + 1e-5, 1 - 1e-5)).mean()
            bc_loss = -log_probs_mean

            # Q loss
            q_loss = -self.network.select('critic')(batch_select['observations'], actions=actor_actions).mean()
            # Actor entropy maximization loss
            entropy_max_loss = (log_probs * self.network.select('alpha')()).mean()

            # Alpha loss
            alpha = self.network.select('alpha')(params=grad_params)
            entropy = -jax.lax.stop_gradient(log_probs).mean()
            alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

            actor_loss = (
                q_loss +
                bc_loss * self.config['alpha'] +
                entropy_max_loss +
                alpha_loss
            )

            return actor_loss, {
                'bc_loss': bc_loss,
                'q_loss': q_loss,
                'entropy_max_loss': entropy_max_loss,
                'alpha_loss': alpha_loss,
                'alpha': alpha,
                'entropy': -log_probs.mean(),
                'actions_mean': actor_actions.mean(),
                'actions_std': actor_actions.std(),
                'actions_max': actor_actions.max(),
                'actions_min': actor_actions.min(),
            }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        # Mark completion transitions as terminals. For single-action policies,
        # this only matters for the critic loss.
        batch['terminals'] = (batch['terminals'].astype(bool) | ~batch['masks'].astype(bool)).astype(jnp.float32)

        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        info['mean_completions'] = (1 - batch['masks']).sum(axis=1).mean()
        info['mean_terminations'] = batch['terminals'].sum(axis=1).mean()

        loss = critic_loss + actor_loss
        return loss, info
        

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        assert observations.shape[-1] == self.config['obs_dim']

        # Observations can be any shape, as long as the last dimension is the
        # observation dimension.
        if self.config['actor_type'] == 'best-of-n':
            batch_dims = observations.shape[:-1]
            observations = observations.reshape((-1, self.config['obs_dim']))
            num_observations = observations.shape[0]
            noises = jax.random.normal(
                rng,
                (
                    (num_observations, self.config['actor_num_samples'], self.config['action_chunk_dim'])
                ),
            )
            observations = repeat(
                observations,
                "obs obs_dim -> obs samples_per_obs obs_dim",
                samples_per_obs=self.config["actor_num_samples"],
            )
            assert observations.shape[:-1] == noises.shape[:-1]

            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            assert actions.shape == (num_observations, self.config['actor_num_samples'], self.config['action_chunk_dim'])
            q_ens = self.network.select('critic')(observations, actions=actions)
            assert q_ens.shape == ((self.config['num_critics'], num_observations, self.config['actor_num_samples']))
            q = reduce(q_ens, 'ensemble batch sample -> batch sample', 'mean')
            actions = actions[jnp.arange(num_observations), jnp.argmax(q, axis=1)].reshape(batch_dims + (self.config['action_chunk_dim'],))

        elif self.config['actor_type'] == 'fql':
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[:-1],  # batch dims
                    self.config['action_chunk_dim'],
                ),
            )
            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        else: # gaussian
            dist = self.network.select('actor')(observations)
            actions = dist.sample(seed=rng)
            actions = jnp.clip(actions, -1, 1)

        return actions
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        batch_dims = noises.shape[:-1]
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full(batch_dims + (1,), i / self.config['flow_steps'])
            vels = self.network.select('actor')(observations, actions, t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        assert ex_observations.ndim == 3, ex_observations.shape
        assert ex_actions.ndim == 3, ex_actions.shape
        assert config['horizon_length'] % (config['action_chunk_size'] * config['action_chunk_eval_interval']) == 0
        assert not (config['actor_type'] == 'gaussian' and config['action_chunk_size'] > 1), "Gaussian actor with action chunking is not supported"

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        config['action_dim'] = ex_actions.shape[-1]
        config['action_chunk_dim'] = config['action_chunk_size'] * config['action_dim']
        config['obs_dim'] = ex_observations.shape[-1]
        batch_size = ex_observations.shape[0]
        assert ex_observations.shape[1] == config['horizon_length'], (ex_observations.shape, config['horizon_length'])

        config['num_eval_chunks_per_seq'] = config['horizon_length'] // (config['action_chunk_eval_interval'] * config['action_chunk_size'])
        print("num_eval_chunks_per_seq:", config['num_eval_chunks_per_seq'])
        print("action_chunk_dim:", config['action_chunk_dim'])

        batch_select = cls.select_obs_and_act_chunks(
            {
                'observations': ex_observations,
                'actions': ex_actions,
            },
            config['action_chunk_size'],
            config['action_chunk_eval_interval'],
        )
        ex_observations = batch_select['observations']
        ex_action_chunks = batch_select['action_chunks']

        config['target_entropy'] = -config['target_entropy_multiplier'] * config['action_chunk_dim']

        # Define networks.
        critic_def = Value(
            hidden_dims=config['critic_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_critics'],
        )
        assert config['actor_type'] in ['best-of-n', 'gaussian', 'fql'], config['actor_type']
        if config['actor_type'] == 'gaussian':
            actor_base_cls = partial(MLP, hidden_dims=config['actor_hidden_dims'], activate_final=True, layer_norm=config['actor_layer_norm'])
            actor_def = TanhNormal(actor_base_cls, config['action_chunk_dim'])
            actor_params = (ex_observations,)
        else:
            actor_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['action_chunk_dim'],
                layer_norm=config['actor_layer_norm'],
            )
            actor_params = (ex_observations, ex_action_chunks, jnp.zeros((batch_size, config['num_eval_chunks_per_seq'], 1)))
        
        # Only used for actor_type=fql
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['action_chunk_dim'],
            layer_norm=config['actor_layer_norm'],
        )
        
        # Define the dual alpha variable
        alpha_def = Temperature(config["init_temp"])

        network_info = dict(
            actor=(actor_def, actor_params),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_action_chunks)),
            critic=(critic_def, (ex_observations, ex_action_chunks)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_action_chunks)),
            alpha=(alpha_def, ()),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        optimizer = optax.adam(learning_rate=config['lr'])

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=optimizer)

        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']


        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


    
def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='lql',
            
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).

            horizon_length=ml_collections.config_dict.placeholder(int), # Will be set

            action_chunk_size=1,
            action_chunk_eval_interval=1, # Interval at which action chunks should have Q-values evaluated.

            # Critic
            critic_hidden_dims=(512, 512, 512, 512),
            num_critics=2,
            layer_norm=True,  # Whether to use layer normalization for the critic.
            hinge_loss_weight=1.0,  # Weight for the lql hinge loss.

            # Actor
            actor_type='best-of-n',
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.

            # best-of-n actor
            actor_num_samples=16, # Number of action samples for actor best-of-n
            flow_steps=10,  # Number of flow steps.

            # fql and gaussian actor
            alpha=0.0, # Behavior cloning weight

            # Gaussian actor
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            init_temp=1.0,

            tau=0.005,  # Target network update rate.
            discount=0.99,  # Discount factor.
            lr=3e-4,  # Learning rate.
            batch_size=256,


        )
    )
    return config
    