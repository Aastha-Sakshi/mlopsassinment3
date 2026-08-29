"""Validate the final demonstration video with ffprobe and FFmpeg."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def black_sections(path: Path) -> list[dict[str, float]]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=2:pix_th=0.06",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    found = []
    for match in re.finditer(
        r"black_start:(?P<start>[\d.]+) black_end:(?P<end>[\d.]+) "
        r"black_duration:(?P<duration>[\d.]+)",
        result.stderr,
    ):
        found.append({name: float(value) for name, value in match.groupdict().items()})
    return found


def validate(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Video is missing or empty: {path}")
    payload = probe(path)
    streams = payload["streams"]
    video = next((stream for stream in streams if stream["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("No video stream found.")
    duration = float(payload["format"]["duration"])
    if not 1 < duration < 300:
        raise ValueError(f"Video duration must be below five minutes, found {duration:.3f}s")
    if (video.get("width"), video.get("height")) != (1920, 1080):
        raise ValueError(f"Expected 1920x1080, found {video.get('width')}x{video.get('height')}")
    black = black_sections(path)
    if black:
        raise ValueError(f"Detected black sections lasting at least two seconds: {black}")
    return {
        "path": str(path.resolve()),
        "duration_seconds": duration,
        "resolution": f"{video['width']}x{video['height']}",
        "video_codec": video["codec_name"],
        "audio_streams": sum(stream["codec_type"] == "audio" for stream in streams),
        "black_sections_over_two_seconds": black,
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?", default=Path("docs/demo.mp4"))
    args = parser.parse_args()
    print(json.dumps(validate(args.video), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
