from typing import Any, Iterable, Callable, Sequence

import multiprocessing as mp
from multiprocessing.connection import Connection

from copy import deepcopy

import numpy as np

from ..env import Env
from .common import step_with_autoreset, get_attribute, info_union


class AsyncVectorEnv:
    parent_conns: tuple[Connection, ...]
    child_conns:  tuple[Connection, ...]

    def __init__(
            self,
            env_fns: Iterable[Callable[[], Env]],
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

        for process in self.processes:
            process.start()


    def handle_errors(self, results: Any, successes: tuple[bool, ...]) -> None:
        if not all(successes):
            raise ValueError(f'Error in reset: {results}')


    def reset(self) -> tuple[list[Any], dict[str, Any]]:
        for conn in self.parent_conns:
            conn.send(('reset', ()))

        results, successes = zip(*[conn.recv() for conn in self.parent_conns])

        self.handle_errors(results, successes)

        obs, infos = map(list, zip(*results))

        if self.copy:
            obs = deepcopy(obs)

        return obs, info_union(infos)


    def step(self, actions: Iterable[Any], *args: Any, **kwargs: Any) -> tuple[list[Any], list[float], list[bool], list[bool], dict[str, Any]]:
        if (isinstance(actions, Sequence) and len(actions) != self.n_envs) or sum(1 for _ in actions) != self.n_envs:
            raise ValueError(f'Number of actions does not match number of environments ({self.n_envs})')


        for conn, action in zip(self.parent_conns, actions):
            conn.send(('step', (action, args, kwargs)))

        results, successes = zip(*[conn.recv() for conn in self.parent_conns])

        self.handle_errors(results, successes)

        obs, rewards, terminated, truncated, infos = map(list, zip(*results))

        if self.copy:
            obs = deepcopy(obs)

        return obs, rewards, terminated, truncated, info_union(infos)


    def render(self) -> None:
        for conn in self.parent_conns:
            conn.send(('render', ()))

        [conn.recv() for conn in self.parent_conns]


    def close(self) -> None:
        for conn in self.parent_conns:
            conn.send(('close', ()))

        for conn in self.parent_conns:
            conn.close()

        for process in self.processes:
            process.join()


    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        for conn in self.parent_conns:
            conn.send(('call', (name, args, kwargs)))

        results, successes = zip(*[conn.recv() for conn in self.parent_conns])

        self.handle_errors(results, successes)

        return tuple(map(list, zip(*results)))


    def _worker(
            self,
            conn: Connection,
            env_fn: Callable[[], Env],
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