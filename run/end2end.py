from pathlib import Path
import argparse

import random
import numpy as np

import torch
import torch.optim as optim

from cpscheduler.algorithms.end2end import End2End
from cpscheduler.environment.wrappers import End2EndStateWrapper
from cpscheduler.policies.end2end import End2EndActor

from cpscheduler.environment.instances import generate_taillard_instance
from cpscheduler.common_envs import JobShopEnv

root = Path().absolute().parent


argparser = argparse.ArgumentParser()
argparser.add_argument('-d', type=int, default=8, help='Dimension of the model')
argparser.add_argument('-l', '--layers', type=int, default=1, help='Number of layers')
argparser.add_argument('--heads', type=int, default=1, help='Number of heads')
argparser.add_argument('--dropout', type=float, default=0., help='Dropout rate')

argparser.add_argument('--save', type=str, default=f'{root}/models/model.pth', help='Path to save the model')
argparser.add_argument('--load', type=str, default=None, help='Path to load the model')
argparser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
argparser.add_argument('--batch-size', type=int, default=128, help='Batch size')
argparser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
argparser.add_argument('--update-steps', type=int, default=16, help='Number of steps per update')
argparser.add_argument('--n-jobs', type=int, default=15, help='Number of jobs')
argparser.add_argument('--n-machines', type=int, default=15, help='Number of machines')
argparser.add_argument('--n-envs', type=int, default=128, help='Number of parallel environments')
argparser.add_argument('--vector-env', type=str, default='async', help='Type of vectorized environment')
argparser.add_argument('--n-future-tasks', type=int, default=3, help='Number of future tasks to consider')
argparser.add_argument('--clip-coef', type=float, default=0.2, help='PPO clip coefficient')
argparser.add_argument('--seed', type=int, default=42, help='Random seed')
argparser.add_argument('--cuda', type=int, nargs='?', const=True, default=False, help='Use CUDA. If a number is passed, it is used as the device index')

argparser.add_argument('--name', type=str, default='experiment', help='Name of the experiment')
argparser.add_argument('--wandb', action='store_true', help='Use wandb for logging')

args = argparser.parse_args()

config = {
    'agent': {
        'model_dim': args.d,
        'layers': args.layers,
        'heads': args.heads,
        'dropout': args.dropout,
    },
    'training': {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'update_steps': args.update_steps,
        'lr': args.lr,
        'clip_coef': args.clip_coef,
    },
    'environment': {
        'n_jobs': args.n_jobs,
        'n_machines': args.n_machines,
        'n_envs': args.n_envs,
        'n_future_tasks': args.n_future_tasks,
    },
    'seed': args.seed,
    'cuda': args.cuda,
}



if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


device = 'cpu'
if torch.cuda.is_available() and args.cuda is not False:
    device = "cuda" if args.cuda is True else f"cuda:{args.cuda}"


actor = End2EndActor(
    args.d, args.heads, args.layers, args.dropout
).to(device)


optimizer = optim.Adam(actor.parameters(), lr=args.lr)

def make_env() -> End2EndStateWrapper:
    instance, metadata = generate_taillard_instance(args.n_jobs, args.n_machines)

    env = JobShopEnv(instance)

    return End2EndStateWrapper(env, args.n_future_tasks)


algorithm = End2End(
    actor, optimizer, make_env,
    args.n_envs, args.n_jobs, args.n_machines,
    args.vector_env, device,
    n_future_tasks=args.n_future_tasks,
    clip_coef=args.clip_coef
)

algorithm.begin_experiment(
    'end2end-makespan',
    args.name,
    root / 'logs',
    args.wandb,
    config
)

try:
    algorithm.learn(
        args.epochs,
        args.update_steps,
        args.batch_size,
        validation_freq=None,
    )

finally:
    torch.save(actor.state_dict(), args.save)