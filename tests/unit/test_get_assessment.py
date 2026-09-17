from uuid import UUID

import pytest

from loadvision.application.get_assessment import GetAssessment
from loadvision.errors import ErrorCode, LoadvisionError
from tests.builders import make_assessment, make_image_record, make_label
from tests.fakes.in_memory_db import InMemoryStore
from tests.port_contracts.unit_of_work import store_assessment


@pytest.mark.unit
def test_assessment_comes_with_its_photo_record_and_label() -> None:
    store = InMemoryStore()
    assessment = make_assessment()
    with store.unit_of_work() as uow:
        store_assessment(uow, assessment)
        uow.labels.save(make_label())
        uow.commit()

    details = GetAssessment(uow_factory=store.unit_of_work)(assessment.id)

    assert details.assessment == assessment
    assert details.image == make_image_record()
    assert details.label == make_label()


@pytest.mark.unit
def test_assessment_without_a_label_has_none() -> None:
    store = InMemoryStore()
    assessment = make_assessment()
    with store.unit_of_work() as uow:
        store_assessment(uow, assessment)
        uow.commit()

    assert GetAssessment(uow_factory=store.unit_of_work)(assessment.id).label is None


@pytest.mark.unit
def test_unknown_assessment_is_not_found() -> None:
    with pytest.raises(LoadvisionError) as raised:
        GetAssessment(uow_factory=InMemoryStore().unit_of_work)(UUID(int=404))
    assert raised.value.code is ErrorCode.ASSESSMENT_NOT_FOUND


@pytest.mark.unit
def test_assessment_without_a_photo_record_is_an_internal_error() -> None:
    store = InMemoryStore()
    # The fake has no foreign keys, so the broken state the database forbids can be built.
    store.assessments[UUID(int=1)] = make_assessment()

    with pytest.raises(LoadvisionError) as raised:
        GetAssessment(uow_factory=store.unit_of_work)(UUID(int=1))
    assert raised.value.code is ErrorCode.INTERNAL_ERROR
