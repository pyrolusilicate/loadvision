"""Load test of the running service: how long an answer takes under parallel clients.

The model alone is measured by `loadvision bench`. This measures everything a client actually
waits for: the XML request, the photo decoding, the model, the database and the answer. Every
request carries a photo of its own, so nothing is served from a stored assessment.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx2
from PIL import Image

NAMESPACE: Final = "urn:loadvision:api:v1"
XML: Final = {"Content-Type": "application/xml; charset=utf-8"}
DEFAULT_URL: Final = "http://localhost:8000"
DEFAULT_PHOTOS: Final = Path("data/test/images")
# Long enough for a photo on a CPU, short enough to notice a service that stopped answering.
TIMEOUT_SECONDS: Final = 120.0

_logger = logging.getLogger("loadtest")


@dataclass(frozen=True, slots=True)
class Answer:
    """What one request cost.

    Attributes:
        seconds: Time from sending the request to reading the answer.
        status: HTTP status, or 0 when nothing came back.
        detail: Error code or message, empty for an answer that arrived as expected.
    """

    seconds: float
    status: int
    detail: str = ""


def unique_photos(folder: Path, count: int, tag: str) -> list[bytes]:
    """Returns `count` photos, each with bytes of its own.

    The service answers a repeated photo from the stored assessment, so a load test that sends
    one photo twice would measure the database and not the model - and a second run of the test
    would measure the first one. One pixel is painted with the run and the number in every copy:
    the picture stays the same, the hash does not.

    Args:
        folder: Folder with the photos to take from; they are reused in a circle.
        count: How many photos to prepare.
        tag: Mark of this run, so two runs never send the same photo.

    Returns:
        JPEG bytes, `count` of them.

    Raises:
        FileNotFoundError: If the folder holds no JPEG.
    """
    sources = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    if not sources:
        raise FileNotFoundError(f"{folder} holds no JPEG to send")
    photos: list[bytes] = []
    for number in range(count):
        with Image.open(sources[number % len(sources)]) as picture:
            copy = picture.convert("RGB")
        # A pixel of its own in the corner: the hash differs, the load on the photo does not.
        mark = int(tag) % 256
        copy.putpixel((number % copy.width, 0), (number % 256, mark, (number + mark) % 256))
        buffer = io.BytesIO()
        copy.save(buffer, "JPEG", quality=92)
        photos.append(buffer.getvalue())
    return photos


def request_body(shipment_id: str, photo: bytes) -> bytes:
    """Returns the XML body that asks the service to assess one photo."""
    image = base64.b64encode(photo).decode()
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<AssessmentRequest xmlns="{NAMESPACE}">'
        f"<ShipmentId>{shipment_id}</ShipmentId><Image>{image}</Image>"
        "</AssessmentRequest>"
    ).encode()


async def assess(client: httpx2.AsyncClient, url: str, body: bytes) -> Answer:
    """Sends one photo and returns what the answer cost."""
    started = time.perf_counter()
    try:
        response = await client.post(f"{url}/api/v1/assessments", content=body, headers=XML)
    except httpx2.HTTPError as error:
        return Answer(time.perf_counter() - started, 0, f"{type(error).__name__}: {error}")
    spent = time.perf_counter() - started
    if response.status_code >= 400:
        return Answer(spent, response.status_code, response.text[:200])
    return Answer(spent, response.status_code)


async def run(url: str, clients: int, requests: int, photos: list[bytes], tag: str) -> list[Answer]:
    """Sends `requests` photos through `clients` parallel clients and returns every answer."""
    queue: asyncio.Queue[int] = asyncio.Queue()
    for number in range(requests):
        queue.put_nowait(number)
    answers: list[Answer] = []

    async def worker() -> None:
        limits = httpx2.Limits(max_connections=2, max_keepalive_connections=2)
        async with httpx2.AsyncClient(timeout=TIMEOUT_SECONDS, limits=limits) as client:
            while True:
                try:
                    number = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                body = request_body(f"LOAD-{tag}-{number:04d}", photos[number])
                answers.append(await assess(client, url, body))

    await asyncio.gather(*(worker() for _ in range(clients)))
    return answers


def report(answers: list[Answer], wall_seconds: float, clients: int) -> dict[str, object]:
    """Turns the answers into the numbers a README quotes."""
    # 201 is a fresh assessment, 200 is the stored answer to a photo the shipment already has:
    # both are answers a client waited for, and the split says which path was measured.
    good = sorted(answer.seconds for answer in answers if answer.status in {200, 201})
    reused = sum(1 for answer in answers if answer.status == 200)
    failures = [answer for answer in answers if answer.status not in {200, 201}]
    quantiles: dict[str, float] = {}
    if good:
        quantiles = {
            "p50_ms": round(statistics.median(good) * 1000, 1),
            "p95_ms": round(good[min(int(len(good) * 0.95), len(good) - 1)] * 1000, 1),
            "max_ms": round(good[-1] * 1000, 1),
        }
    return {
        "clients": clients,
        "requests": len(answers),
        "answered": len(good),
        "reused": reused,
        "failed": len(failures),
        "wall_seconds": round(wall_seconds, 1),
        "photos_per_second": round(len(good) / wall_seconds, 2) if wall_seconds else 0.0,
        **quantiles,
        "failures": [f"{answer.status}: {answer.detail}" for answer in failures[:3]],
    }


def main() -> int:
    """Runs the load test and prints the report as a table and as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="service address (default: %(default)s)")
    parser.add_argument(
        "--clients", type=int, default=4, help="parallel clients (default: %(default)s)"
    )
    parser.add_argument(
        "--requests", type=int, default=20, help="photos to send (default: %(default)s)"
    )
    parser.add_argument(
        "--photos", type=Path, default=DEFAULT_PHOTOS, help="folder to take photos from"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the report as JSON here")
    parser.add_argument(
        "--tag",
        default=None,
        help="mark of the run; the same mark sends the same photos again, which measures the "
        "path of a photo the service already knows (default: the time now)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # One line per request says nothing about the load; the numbers below do.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    tag = args.tag or datetime.now(UTC).strftime("%H%M%S")
    photos = unique_photos(args.photos, args.requests, tag)
    _logger.info("%d photos, %d clients, %s", args.requests, args.clients, args.url)
    started = time.perf_counter()
    answers = asyncio.run(run(args.url, args.clients, args.requests, photos, tag))
    numbers = report(answers, time.perf_counter() - started, args.clients)

    for name, value in numbers.items():
        if name != "failures":
            _logger.info("  %s: %s", name, value)
    for failure in numbers["failures"] if isinstance(numbers["failures"], list) else []:
        _logger.warning("  failure: %s", failure)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(numbers, indent=2, ensure_ascii=False) + "\n")
        _logger.info("  written: %s", args.out)
    return 0 if numbers["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
