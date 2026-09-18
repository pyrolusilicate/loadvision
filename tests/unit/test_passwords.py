import pytest

from loadvision.application.passwords import (
    MIN_PASSWORD_LENGTH,
    check_password,
    hash_password,
    verify_password,
)

PASSWORD = "correct horse battery"


@pytest.fixture(scope="module")
def stored() -> str:
    return hash_password(PASSWORD)


@pytest.mark.unit
def test_the_right_password_matches_and_a_wrong_one_does_not(stored: str) -> None:
    assert verify_password(PASSWORD, stored) is True
    assert verify_password(PASSWORD + "!", stored) is False


@pytest.mark.unit
def test_the_same_password_gets_a_new_salt_every_time(stored: str) -> None:
    again = hash_password(PASSWORD)
    assert again != stored
    assert stored.startswith("scrypt$")


@pytest.mark.unit
@pytest.mark.parametrize(
    "malformed", ["", "scrypt", "scrypt$$", "bcrypt$00$00", "scrypt$zz$00", "scrypt$00$zz"]
)
def test_a_malformed_hash_matches_nothing(malformed: str) -> None:
    assert verify_password(PASSWORD, malformed) is False


@pytest.mark.unit
def test_a_short_password_is_refused() -> None:
    check_password("x" * MIN_PASSWORD_LENGTH)
    with pytest.raises(ValueError, match="at least"):
        check_password("x" * (MIN_PASSWORD_LENGTH - 1))
