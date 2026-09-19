import pytest
import torch

from loadvision.ml.devices import resolve_device


@pytest.mark.unit
@pytest.mark.parametrize(("gpu", "device"), [(True, "cuda"), (False, "cpu")])
def test_auto_takes_the_gpu_when_torch_sees_one(
    monkeypatch: pytest.MonkeyPatch, gpu: bool, device: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)

    assert resolve_device("auto") == device


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "cuda:1"])
def test_a_named_device_is_kept_as_given(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device(device) == device
