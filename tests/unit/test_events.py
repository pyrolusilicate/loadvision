import re

import pytest

from loadvision.observability.events import Event


@pytest.mark.unit
def test_event_names_are_unique() -> None:
    # A repeated value turns the second member into a silent alias that iteration skips.
    assert len(Event.__members__) == len(Event)


@pytest.mark.unit
@pytest.mark.parametrize("event", list(Event))
def test_event_name_is_lowercase_subject_dot_action(event: Event) -> None:
    assert re.fullmatch(r"[a-z]+\.[a-z_]+", event.value)
