from typing import Any, Iterable, Callable, Sequence, SupportsFloat, TypeVar

import multiprocessing as mp
from multiprocessing.connection import Connection

from copy import deepcopy

from enum import Enum

from .common import Env, VectorEnv, step_with_autoreset, get_attribute, info_union

class EnvStatus(Enum):
    READY = 0
    RUNNING = 1
    CLOSED = 2


_Obs = TypeVar('_Obs')
_Act = TypeVar('_Act')
class AsyncVectorEnv(VectorEnv[_Obs, _Act]):
    parent_conns: tuple[Connection, ...]
    child_conns:  tuple[Connection, ...]

    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env[_Obs, _Act]]],
            copy      : bool = False,
            auto_reset: bool = True,
            daemon    : bool = True,
        ):
        self.env_fns    = env_fns
        self.copy       = copy

        self.n_envs = sum(1 for _ in env_fns)

        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(self.n_envs)])

        self.processes = [
            mp.Process(target=self._worker, args=(
                child_conn,
                env_fn,
                auto_reset
            ), daemon=daemon)
            for child_conn, env_fn in zip(self.child_conns, env_fns)
        ]

        self.status = EnvStatus.READY

        for process in self.processes:
            process.start()

        self.status = EnvStatus.RUNNING


    # TODO: fix this to handle multiple errors
    def handle_errors(
            self,
            origin: str,
            results: Sequence[Exception],
            successes: tuple[bool, ...],
        ) -> None:

        if not all(successes):
            import traceback

            envs_with_errors = [i for i, success in enumerate(successes) if not success]

            exception_group = ExceptionGroup(
                f"Error during {origin} in {len(envs_with_errors)} environments out of {self.n_envs}:\n",
                [results[i] for i in envs_with_errors]
            )

            raise exception_group from results[envs_with_errors[0]]


    def reset(self) -> tuple[list[_Obs], dict[str, Any]]:
        for conn in self.parent_conns:
            conn.send(('reset', ()))

        results, successes = zip(*[conn.recv() for conn in self.parent_conns])

        self.handle_errors('reset', results, successes)

        obs, infos = map(list, zip(*results))

        if self.copy:
            obs = deepcopy(obs)

        return obs, info_union(infos)


    def step(self, actions: Iterable[_Act], *args: Any, **kwargs: Any) -> tuple[list[_Obs], list[SupportsFloat], list[bool], list[bool], dict[str, Any]]:
        if sum(1 for _ in actions) != self.n_envs:
            raise ValueError(f'Number of actions does not match number of environments ({self.n_envs})')


        for conn, action in zip(self.parent_conns, actions):
            conn.send(('step', (action, args, kwargs)))

        results, successes = zip(*[conn.recv() for conn in self.parent_conns])

        self.handle_errors('step', results, successes)

        obs, rewards, terminated, truncated, infos = map(list, zip(*results))

        if self.copy:
            obs = deepcopy(obs)

        return obs, rewards, terminated, truncated, info_union(infos)


    def render(self) -> None:
        for conn in self.parent_conns:
            conn.send(('render', ()))

        [conn.recv() for conn in self.parent_conns]


    def close(self) -> None:
        if self.status == EnvStatus.CLOSED:
            return

        for conn in self.parent_conns:
            conn.send(('close', ()))

        for conn in self.parent_conns:
            conn.close()

        for process in self.processes:
            process.join()

        self.status = EnvStatus.CLOSED


    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        for conn in self.parent_conns:
            conn.send(('call', (name, args, kwargs)))

        results, successes = zip(*[conn.recv() for conn in self.parent_conns])

        self.handle_errors(name, results, successes)

        return tuple(map(list, zip(*results)))


    def _worker(
            self,
            conn: Connection,
            env_fn: Callable[[], Env[_Obs, _Act]],
            auto_reset: bool
        ) -> None:
        env = env_fn()

        command: str
        data: tuple[Any, ...]
        try:
            while True:
                command, data = conn.recv()

                match command:
                    case 'reset':
                        obs, info = env.reset()

                        conn.send(((obs, info), True))

                    case 'step':
                        action, args, kwargs = data

                        if auto_reset:
                            obs, reward, terminated, truncated, info = step_with_autoreset(env, action, *args, **kwargs)

                        else:
                            obs, reward, terminated, truncated, info = env.step(action, *args, **kwargs)

                        conn.send(((obs, reward, terminated, truncated, info), True))

                    case 'render':
                        env.render()
                        conn.send((None, True))

                    case 'close':
                        conn.close()
                        break

                    case 'call':
                        name, args, kwargs = data

                        if name in ["reset", "step", "close", "render"]:
                            raise ValueError(f'Trying to call function `{name}` with `call`, use `{name}` directly instead.')


                        result = get_attribute(env, name, *args, **kwargs)

                        conn.send((result, True))

                    case _:
                        raise ValueError(f'Unknown command: {command}')


        except Exception as e:
            conn.send((e, False))
        

    def __del__(self) -> None:
        self.close()