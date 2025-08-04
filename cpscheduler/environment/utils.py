"""
utils.py

This module provides utility functions for the environment and other modules.
"""

from typing import Any, TypeVar, overload, Generic, Callable
from collections.abc import Iterable, Iterator
from typing_extensions import TypeIs

from collections import deque

from ._common import TASK_ID, Int

_T = TypeVar("_T", bound=Any)


@overload
def convert_to_list(array: Any, dtype: type[_T]) -> list[_T]: ...
@overload
def convert_to_list(array: Iterable[_T], dtype: None = ...) -> list[_T]: ...
@overload
def convert_to_list(array: Any, dtype: None = ...) -> list[Any]: ...


def convert_to_list(array: Iterable[Any], dtype: type[Any] | None = None) -> list[Any]:
    """
    Convert an iterable to a list. If a dtype is provided, the elements of the list will be casted
    to that type.

    Parameters
    ----------
    array: Iterable
        The iterable to be converted to a list.

    dtype: type, optional
        The type to which the elements of the list will be casted to.

    Returns
    -------
    list
        A list containing the elements of the iterable.
    """

    if hasattr(array, "tolist"):
        array = getattr(array, "tolist")()

    try:
        if dtype is None:
            return array if isinstance(array, list) else list(array)

        return [dtype(item) for item in array]

    # If the iterable is not a collection, it will raise a TypeError
    except TypeError:
        return [array] if dtype is None else [dtype(array)]


def is_iterable_type(
    obj: Any, dtype: type[_T], lazy: bool = True
) -> TypeIs[Iterable[_T]]:
    """
    Returns whether the object is an iterable containing elements of the specified type.

    Parameters
    ----------
    obj: Any
        The object to be checked.

    dtype: type
        The type of the elements of the iterable.

    lazy: bool, optional
        If True, the type checking will only check the first element of the iterable, the
        user is responsible for ensuring that all elements are of the specified type.
        In a compiled environment, the lazy check is preferred.

    Returns
    -------
    bool
        Whether the object is an iterable containing elements of the specified type.
    """
    try:
        if lazy:
            first_item = next(iter(obj))
            return isinstance(first_item, dtype)

        return all(isinstance(item, dtype) for item in obj)

    except StopIteration:
        # If the iterable is empty, we consider it to be of the specified type
        return True

    except TypeError:
        # If the iterable is not a collection, it will raise a TypeError
        return False


def is_iterable_int(obj: Any, lazy: bool = True) -> TypeIs[Iterable[Int]]:
    try:
        if lazy:
            first_item = next(iter(obj))
            return isinstance(first_item, Int)

        return all(isinstance(item, Int) for item in obj)

    # except StopIteration:
    #     # If the iterable is empty, we consider it to be of the specified type
    #     return True

    except TypeError:
        # If the iterable is not a collection, it will raise a TypeError
        return False


def topological_sort(
    precedence_map: dict[TASK_ID, list[TASK_ID]], n_tasks: TASK_ID
) -> list[TASK_ID]:
    """
    Perform a topological sort on a directed acyclic graph.

    Parameters
    ----------
    precedence_map: dict
        A dictionary containing the precedence relationships between the tasks.

    in_degree: list
        A list containing the in-degree of each task.

    Returns
    -------
    list
        A list containing the tasks in topological order
    """
    in_degree = [0] * n_tasks
    for children in precedence_map.values():
        for child in children:
            in_degree[child] += 1

    queue = deque([task for task, degree in enumerate(in_degree) if degree == 0])

    topological_order: list[int] = []

    while queue:
        vertex = queue.popleft()

        if vertex not in precedence_map or not precedence_map[vertex]:
            continue

        topological_order.append(vertex)

        for child in precedence_map[vertex]:
            in_degree[child] -= 1

            if in_degree[child] == 0:
                queue.append(child)

    return topological_order


def binary_search(
    array: list[_T],
    target: _T,
    left: int = 0,
    right: int = -1,
) -> int:
    """
    Perform a binary search on a sorted array.

    Parameters
    ----------
    array: list
        The sorted array to be searched.

    target: Any
        The target value to be searched for.

    left: int, optional
        The left index of the search interval.

    right: int, optional
        The right index of the search interval.

    Returns
    -------
    int
        The index of the inclusion of the value in the array.
    """

    if right < 0:
        right = len(array) + right

    while left <= right:
        mid = (left + right) // 2

        if array[mid] == target:
            return mid

        if array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return left

_K = TypeVar('_K', bound=Any)
class IndexedHeap(Generic[_T, _K]):
    _data: list[tuple[_K, _T]]
    _pos: dict[_T, int]
    _keyfn: Callable[[_T], _K] | None

    def __init__(
        self,
        iterable: Iterable[_T] = (),
        key: Callable[[_T], _K] | Iterable[_K] = lambda x: x,
    ) -> None:
        if callable(key):
            self._keyfn = key
            self._data = [
                (self._keyfn(item), item) for item in iterable
            ]

        else:
            self._keyfn = None
            self._data = [
                (k, item) for k, item in zip(key, iterable)
            ]

        self._pos = {item: idx for idx, (_, item) in enumerate(self._data)}

        self._restore()

    def __repr__(self) -> str:
        return f"IndexedHeap({self._data})"

    def _sift_up(self, idx: int) -> None:
        key, item = self._data[idx]

        while idx > 0:
            parent = (idx - 1) // 2
            if self._data[parent] <= self._data[idx]:
                break

            self._data[idx], self._data[parent] = self._data[parent], self._data[idx]
            self._pos[self._data[idx][1]] = idx
            idx = parent

        self._data[idx] = (key, item)
        self._pos[item] = idx

    def _sift_down(self, idx: int) -> None:
        size = len(self._data)
        key, item = self._data[idx]

        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx

            if left < size and self._data[left] < self._data[smallest]:
                smallest = left

            if right < size and self._data[right] < self._data[smallest]:
                smallest = right

            if smallest == idx:
                break

            self._data[idx], self._data[smallest] = self._data[smallest], self._data[idx]
            self._pos[self._data[idx][1]] = idx
            idx = smallest
        
        self._data[idx] = (key, item)
        self._pos[item] = idx

    def _restore(self) -> None:
        if not self._data:
            return

        for idx in range(len(self._data) // 2, -1, -1):
            self._sift_down(idx)

    def insert(self, item: _T, key: _K | None = None) -> None:
        if key is None:
            if self._keyfn is not None:
                key = self._keyfn(item)

            else:
                raise ValueError("Key must be provided or a key function must be set.")

        idx = len(self._data)
        self._data.append((key, item))

        self._pos[item] = idx
        self._sift_up(idx)

    def extend(self, items: Iterable[_T], keys: Iterable[_K] | None) -> None:
        if keys is None:
            if self._keyfn is not None:
                keys = [self._keyfn(item) for item in items]
            
            else:
                raise ValueError("Keys must be provided or a key function must be set.")

        self._pos.update(
            {item: len(self._data) + idx for idx, item in enumerate(items)}
        )
        self._data.extend(zip(keys, items))
        self._restore()

    def remove(self, item: _T) -> None:
        removal_idx = self._pos.pop(item)
        last_key, last_item = self._data.pop()

        if removal_idx == len(self._data):
            return

        self._data[removal_idx] = last_key, last_item
        self._pos[last_item] = removal_idx

        self._sift_down(removal_idx)
        self._sift_up(removal_idx)

    def pop(self) -> _T:
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        _, item = self._data.pop()
        self._pos.pop(item, None)

        if self._data:
            self._sift_down(0)

        return item

    def peek(self) -> tuple[_T, _K]:
        if not self._data:
            raise IndexError("Peek from an empty IndexedHeap.")

        key, item = self._data[0]
        return item, key

    def update(self, item: _T, key: _K | None) -> None:
        if item in self._pos:
            self.remove(item)

        if key is None:
            if self._keyfn is not None:
                key = self._keyfn(item)
            else:
                raise ValueError("Key must be provided or a key function must be set.")

        self.insert(item, key)

    def update_batch(self, items: Iterable[_T], keys: Iterable[_K] | None) -> None:
        # Convert to list for indexed access
        if keys is None:
            if self._keyfn is not None:
                keys = [self._keyfn(item) for item in items]
            else:
                raise ValueError("Keys must be provided or a key function must be set.")

        for item, key in zip(items, keys):
            if item not in self._pos:
                continue  # Ignore unknown items

            idx = self._pos[item]
            self._data[idx] = (key, item)
        
        self._restore()

    def __iter__(self) -> Iterator[_T]:
        return (item for _, item in self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: _T) -> bool:
        return item in self._pos
    
    def clear(self) -> None:
        self._data.clear()
        self._pos.clear()

    def __bool__(self) -> bool:
        return bool(self._data)

    def ordered(self) -> Iterator[tuple[_T, _K]]:
        """Yield items in order of increasing key without modifying the heap."""
        import heapq

        if not self._data:
            return

        heap: list[tuple[_K, int]] = [(self._data[0][0], 0)]
        visited = set()

        while heap:
            key, idx = heapq.heappop(heap)
            _, item = self._data[idx]
            yield item, key
            visited.add(idx)

            for child in (2 * idx + 1, 2 * idx + 2):
                if child < len(self._data) and child not in visited:
                    heapq.heappush(heap, (self._data[child][0], child))
    
    def get_key(self, item: _T) -> tuple[_K, bool]:
        "Get the key of an item, returning (key, found)."
        if item in self._pos:
            idx = self._pos[item]
            return self._data[idx][0], True
    
        if self._keyfn is not None:
            return self._keyfn(item), False
    
        raise KeyError(f"Item {item} not found in IndexedHeap and no key function provided.")
