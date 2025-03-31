from pathlib import Path

from time import perf_counter

import numpy as np

from cpscheduler.environment import SchedulingCPEnv, JobShopSetup
from cpscheduler.instances import read_jsp_instance


benchmark_times = {
    'dmu10' : 0.4,
    'dmu20' : 0.8,
    'dmu30' : 1.4,
    'dmu40' : 2.1,
    'dmu50' : 0.4,
    'dmu60' : 0.8,
    'dmu70' : 1.6,
    'dmu80' : 2.2,
    'la10'  : 0.05,
    'la20'  : 0.05,
    'la30'  : 0.18,
    'la40'  : 0.18,
    'orb10' : 0.05,
    'swv10' : 0.3,
    'swv20' : 1.,
    'ta10'  : 0.16,
    'ta20'  : 0.3,
    'ta30'  : 0.4,
    'ta40'  : 0.6,
    'ta50'  : 0.8,
    'ta60'  : 1.6,
    'ta70'  : 2.,
    'ta80'  : 7.8,
    "kopt_ops10000_m100_1" : 3 * 60
}

root = Path(__file__).parent


def test_speed() -> None:
    print(f"{'| Instance': <9} | {'Time took': <10} | Benchmark | Speedup")
    print('| ' + "-" * 8 + ' | ' + "-" * 10 + ' | ' + "-" * 14 + ' | ' + "-" * 7)

    for instance_name, time in benchmark_times.items():
        instance_path = root / 'instances/jobshop' / f'{instance_name}.txt'

        instance, metadata = read_jsp_instance(instance_path)

        env = SchedulingCPEnv(JobShopSetup())
        env.set_instance(instance, jobs='job')

        obs, info = env.reset()

        # _, action, _, __ = env.get_cp_solution(timelimit=2)

        task_order: list[int] = np.argsort(obs['processing_time']).tolist()

        action = [
            ("submit", task_id) for task_id in task_order
        ]

        tick = perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        measured_time = perf_counter() - tick

        speedup_factor = time / measured_time


        print(
            f"|  {instance_name: <7} |   {measured_time:.2f} s   |    {time:.2f} s    | {speedup_factor:5.2%} |"
        )


if __name__ == '__main__':
    test_speed()