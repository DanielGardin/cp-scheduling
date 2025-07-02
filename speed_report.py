from pathlib import Path

from time import perf_counter
from prettytable import PrettyTable

from cpscheduler import SchedulingEnv, JobShopSetup
from cpscheduler.instances import read_jsp_instance
from cpscheduler.heuristics import ShortestProcessingTime


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
    # "kopt_ops10000_m100_1" : 3 * 60
}

root = Path(__file__).parent


def test_speed(n: int = 1) -> None:
    table = PrettyTable(["Instance", "Benchmark", "Time took", "Speedup"])

    for instance_name, bench_time in benchmark_times.items():
        instance_path = root / 'instances/jobshop' / f'{instance_name}.txt'

        instance, metadata = read_jsp_instance(instance_path)

        env = SchedulingEnv(JobShopSetup())
        env.set_instance(instance, processing_times='processing_time')
        spt_agent = ShortestProcessingTime()

        times = []
        for _ in range(n):
            obs, info = env.reset()
            action = spt_agent(obs)

            tick = perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            times.append(perf_counter() - tick)

        speedups = [bench_time / t - 1 for t in times]

        mean_time = sum(times) / len(times)
        std_time  = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

        mean_speedup = sum(speedups) / len(speedups)
        std_speedup  = (sum((s - mean_speedup) ** 2 for s in speedups) / len(speedups)) ** 0.5


        table.add_row([
            instance_name,
            f"{bench_time:.2f} s",
            f"{mean_time:.3f} s ± {std_time:.3f} s"   if std_time > 0    else f"{mean_time:.3f} s",
            f"{mean_speedup:.2%} ± {std_speedup:.2%}" if std_speedup > 0 else f"{mean_speedup:.2%}"
        ])

    table._set_markdown_style()
    print(table)


if __name__ == '__main__':
    import tyro

    tyro.cli(test_speed, description="Test the speed of the Shortest Processing Time heuristic on various job shop instances.")

    # test_speed()