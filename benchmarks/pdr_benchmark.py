from pathlib import Path

import pandas as pd
from tqdm import tqdm

from time import perf_counter

from cpscheduler.environment import SchedulingCPEnv, JobShopSetup
from cpscheduler.instances import read_jsp_instance
from cpscheduler.policies.heuristics import ShortestProcessingTime, MostOperationsRemaining, MostWorkRemaining, PriorityDispatchingRule

root = Path(__file__).parent.parent

heuristics: dict[str, PriorityDispatchingRule] = {
    "SPT"  : ShortestProcessingTime(),
    "MOPNR": MostOperationsRemaining(),
    "MWKR" : MostWorkRemaining()
}

datasets = {
    "taillard"     : "ta",
    "demirkol"     : "dmu",
    "lawrence"     : "la",
    "applegate"    : "orb",
    "storer"       : "swv",
    "yamada"       : "yn",
    "large-TA"     : "lta",
    "known optimal": "kopt",
}

index_names = ['dataset', 'size', 'name']
column_names = [(heuristic_name, metric) for heuristic_name in heuristics.keys() for metric in ('makespan', 'time')]


data: list[list[float]] = []
indices: list[tuple[str, tuple[int, int], str]] = []

for dataset in datasets:
    dataset_code = datasets[dataset]

    all_instances = tqdm(sorted([
        path for path in (root / 'instances/jobshop').glob(f'{dataset_code}*.txt')
    ]))

    if not all_instances:
        continue

    for instance_path in all_instances:
        instance, metadata = read_jsp_instance(instance_path)

        instance_size = (int(metadata['n_jobs']), int(metadata['n_machines']))
        instance_name = instance_path.stem

        all_instances.set_description(f"Processing instance {instance_name}")

        env = SchedulingCPEnv(JobShopSetup())
        env.set_instance(instance, processing_times="processing_time" ,job_ids='job')

        heuristic_results: list[float] = [0] * len(column_names)

        for i, heuristic in enumerate(heuristics.values()):
            tick = perf_counter()

            obs, info = env.reset()
            action = heuristic(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            tock = perf_counter()

            heuristic_results[2*i:2*(i+1)] = (info['current_time'], tock - tick)

        indices.append((dataset, instance_size, instance_name))
        data.append(heuristic_results)

    dict_data = {
        'index'      : indices,
        'columns'    : column_names,
        'data'       : data,
        'index_names': index_names,
        'column_names': ['heuristic', 'metric']
    }

    results = pd.DataFrame.from_dict(dict_data, orient='tight')
    results.to_csv(root / 'benchmarks' / 'pdr_benchmark.csv')