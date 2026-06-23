from collections.abc import Iterator

from cpscheduler.environment.des.base import EventID, ScheduledEvent


class EventQueue:
    events: list[ScheduledEvent]
    index_map: dict[EventID, int]

    def __init__(self) -> None:
        self.events = []
        self.index_map = {}

    def __bool__(self) -> bool:
        return bool(self.events)

    def __contains__(self, event_id: EventID) -> bool:
        return event_id in self.index_map

    def __iter__(self) -> Iterator[ScheduledEvent]:
        return iter(sorted(self.events))

    def __len__(self) -> int:
        return len(self.events)

    def get(self, event_id: EventID) -> ScheduledEvent:
        return self.events[self.index_map[event_id]]

    def push(self, entry: ScheduledEvent) -> None:
        pos = len(self.events)

        self.events.append(entry)
        self.index_map[entry.event_id] = pos

    def extend(self, entries: list[ScheduledEvent]) -> None:
        pos = len(self.events)

        self.events.extend(entries)
        for idx, entry in enumerate(entries, start=pos):
            self.index_map[entry.event_id] = idx

    def remove(self, event_id: EventID) -> None:
        if event_id not in self.index_map:
            raise KeyError(f"Event {event_id} not found in heap.")

        pos = self.index_map[event_id]
        last_pos = len(self.events) - 1

        removing_event = self.events[pos]
        removing_event.time = -1

        if pos != last_pos:
            last_event = self.events[last_pos]

            self.events[pos], self.events[last_pos] = last_event, removing_event
            self.index_map[last_event.event_id] = pos

        self.events.pop()
        del self.index_map[event_id]
