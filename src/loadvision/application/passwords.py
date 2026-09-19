"""Password hashes of operator accounts: scrypt from the standard library."""

import hashlib
import hmac
import secrets
from typing import Final

MIN_PASSWORD_LENGTH: Final = 10

_SCHEME: Final = "scrypt"
# About 0.1 s per hash on a laptop: slow for someone guessing, unnoticeable for one sign-in.
_COST: Final = 2**14
_BLOCK_SIZE: Final = 8
_PARALLELISM: Final = 1
_SALT_BYTES: Final = 16
_KEY_BYTES: Final = 32


def check_password(password: str) -> None:
    """Refuses a password too short to be worth hashing.

    Raises:
        ValueError: If the password is shorter than `MIN_PASSWORD_LENGTH`.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must have at least {MIN_PASSWORD_LENGTH} characters")


def hash_password(password: str) -> str:
    """Returns the hash to store, ``scrypt$<salt>$<key>`` in hex; a new salt every time."""
    salt = secrets.token_bytes(_SALT_BYTES)
    return f"{_SCHEME}${salt.hex()}${_derive(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Whether the password matches the stored hash; a malformed hash matches nothing."""
    scheme, _, rest = stored.partition("$")
    salt_hex, _, key_hex = rest.partition("$")
    if scheme != _SCHEME or not salt_hex or not key_hex:
        return False
    try:
        salt, key = bytes.fromhex(salt_hex), bytes.fromhex(key_hex)
    except ValueError:
        return False
    # Constant time, so the comparison does not tell how many bytes were right.
    return hmac.compare_digest(_derive(password, salt), key)


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_COST,
        r=_BLOCK_SIZE,
        p=_PARALLELISM,
        dklen=_KEY_BYTES,
    )
