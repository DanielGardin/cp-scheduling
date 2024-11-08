from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cpscheduler.common_envs import JobShopEnv
from cpscheduler.policies.heuristics import ShortestProcessingTime, MostOperationsRemaining, MostWorkRemaining
from cpscheduler.utils import read_instance

import re

root = Path(__file__).parent.parent

heuristics = {
    "SPT"  : ShortestProcessingTime(),
    "MOPNR": MostOperationsRemaining(),
    "MWKR" : MostWorkRemaining()
}


columns = ['n_jobs', 'n_machines', *list(heuristics)]

results = pd.DataFrame(columns=columns)



all_instances = tqdm(sorted([
    path for path in (root / 'instances').glob('ta*.txt') if re.search(r'ta[0-9]+', path.stem)
]))


for instance_path in all_instances:
    instance, metadata = read_instance(instance_path)

    instance_name = instance_path.stem

    all_instances.set_description(f"Processing instance {instance_name}")

    env = JobShopEnv(instance, 'processing_time', dataframe_obs=True)

    heuristic_results = [metadata['n_jobs'], metadata['n_machines']] + [0] * len(heuristics)

    for i, heuristic in enumerate(heuristics.values()):
        obs, info = env.reset()

        action = heuristic(obs)
        obs, reward, terminated, truncated, info = env.step(action, enforce_order=False)

        heuristic_results[2+i] = info['current_time']

    results.loc[instance_name] = heuristic_results

results.to_csv(root / 'benchmarks' / 'taillard_benchmark.csv')