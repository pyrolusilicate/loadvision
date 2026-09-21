import pytest
import torch

from loadvision.ml.models.heads import (
    BIN_COUNT,
    Bins,
    Occupancy,
    Plain,
    bin_centres,
    make_head,
)

WIDTH = 16
PATCHES = 12
PHOTOS = 4
START = 0.42


def tokens() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(PHOTOS, PATCHES, WIDTH)


def vectors() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(PHOTOS, WIDTH)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["linear", "mlp", "occupancy", "bins"])
def test_every_head_starts_at_the_mean_label(kind: str) -> None:
    head = make_head(kind, WIDTH, START)
    features = tokens() if head.wants_tokens else vectors()

    answers = head(features).detach()

    assert answers.shape == (PHOTOS,)
    assert answers == pytest.approx(torch.full((PHOTOS,), START), abs=0.02)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["linear", "mlp", "occupancy", "bins"])
def test_training_a_head_moves_it_towards_the_labels(kind: str) -> None:
    head = make_head(kind, WIDTH, START)
    features = tokens() if head.wants_tokens else vectors()
    targets = torch.tensor([0.0, 0.3, 0.7, 1.0])
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(head.parameters(), lr=0.1)
    before = float(torch.abs(head(features).detach() - targets).mean())

    for _ in range(60):
        loss = head.loss(features, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert float(torch.abs(head(features).detach() - targets).mean()) < before


@pytest.mark.unit
def test_the_occupancy_answer_is_the_weighted_share_of_taken_patches() -> None:
    head = Occupancy(WIDTH, START)
    features = tokens()

    weight, taken = (part.detach() for part in head.maps(features))
    answer = head(features).detach()

    assert weight.sum(dim=1) == pytest.approx(torch.ones(PHOTOS), abs=1e-5)
    assert torch.all((taken >= 0) & (taken <= 1))
    assert answer == pytest.approx((weight * taken).sum(dim=1))


@pytest.mark.unit
def test_the_occupancy_answer_cannot_leave_the_possible_range() -> None:
    head = Occupancy(WIDTH, START)
    with torch.no_grad():
        head.taken.weight.fill_(50.0)
        head.where.weight.fill_(50.0)

    answers = head(tokens() * 100).detach()

    assert torch.all((answers >= 0) & (answers <= 1))


@pytest.mark.unit
def test_the_bins_head_answers_the_mean_of_its_own_distribution() -> None:
    head = Bins(WIDTH, START)
    features = vectors()

    shares = torch.softmax(head.layer(features), dim=1).detach()

    assert shares.shape == (PHOTOS, BIN_COUNT)
    assert head(features).detach() == pytest.approx(shares @ bin_centres(), abs=1e-5)


@pytest.mark.unit
def test_the_bins_loss_is_lowest_when_the_distribution_sits_on_the_label() -> None:
    head = Bins(WIDTH, START)
    features = torch.zeros(1, WIDTH)
    label = torch.tensor([0.5])

    with torch.no_grad():
        head.layer.bias.copy_(-0.5 * ((bin_centres() - 0.5) / 0.075) ** 2)
    on_label = float(head.loss(features, label).detach())
    with torch.no_grad():
        head.layer.bias.copy_(-0.5 * ((bin_centres() - 0.9) / 0.075) ** 2)
    off_label = float(head.loss(features, label).detach())

    assert on_label < off_label


@pytest.mark.unit
def test_the_linear_head_keeps_the_weights_where_the_package_writer_looks() -> None:
    head = Plain(WIDTH, START)

    assert head.weight.shape == (1, WIDTH)
    assert float(head.bias.detach()) == pytest.approx(START)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"kind": "nothing"}, "head"), ({"kind": "linear", "loss": "mse"}, "loss")],
)
def test_an_unknown_head_or_loss_is_refused(kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_head(width=WIDTH, start=START, **kwargs)  # type: ignore[arg-type]
