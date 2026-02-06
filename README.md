# Long-Horizon Q-Learning: Accurate Value Learning via N-Step Inequalities

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
```bash
uv sync                # CPU
uv sync --extra cuda   # GPU
```

## Run

### Best-of-N
```
# LQL
python main.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=4 --agent=lql/agents/lql.py --agent.batch_size=64 --agent.actor_type=best-of-n --agent.actor_num_samples=16


# TD-n
python main.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=4 --agent=lql/agents/tdn.py --agent.actor_type=best-of-n --agent.actor_num_samples=16


# TD
python main.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=1 --agent=lql/agents/tdn.py --agent.actor_type=best-of-n --agent.actor_num_samples=16

# (or, equivalently:)
python main.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=1 --agent=lql/agents/lql.py --agent.batch_size=256 --agent.actor_type=best-of-n --agent.actor_num_samples=16
```

### FQL

See Best-of-N, using `--agent.actor_type=fql` and setting `--agent.alpha`.

### Gaussian
```
# LQL
python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=4 --agent=lql/agents/lql.py --agent.batch_size=64 --agent.actor_type=gaussian


# TD-n
python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=4 --agent=lql/agents/tdn.py --agent.actor_type=gaussian


# TD
python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=1 --agent=lql/agents/tdn.py --agent.actor_type=gaussian
```

### Action chunking
```
# LQL
python main.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=20 --agent=lql/agents/lql.py --agent.batch_size=64 --agent.actor_type=fql --agent.action_chunk_size=5

# TD
python main.py --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=5 --agent=lql/agents/lql.py --agent.actor_type=fql --agent.action_chunk_size=5
```


## Acknowledgments

Thank you to Qiyang Li, Zhiyuan Zhou, and Sergey Levine, who authored
https://github.com/ColinQiyangLi/qc, which this repo was cloned from.