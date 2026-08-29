"""Record the deterministic M1-M5 assignment demonstration with Playwright."""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1920, 1080
API = "http://127.0.0.1:8000"
MLFLOW = "http://127.0.0.1:5000"
REPOSITORY = "https://github.com/Aastha-Sakshi/mlopsassinment3"
PACKAGE = f"{REPOSITORY}/pkgs/container/cats-dogs-api"
BASELINE_RUN_ID = "462a190825b548c7b6d5600724b42cca"
TARGET_SECONDS = 260

TIMELINE: list[tuple[float, str]] = []
STARTED = 0.0

THEME = """
<style>
* { box-sizing: border-box; }
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; }
body { background: #0d1321; color: #e7edf8; font-family: 'Segoe UI', Arial, sans-serif; }
.wrap { width: 100%; height: 100%; padding: 70px 92px; }
.eyebrow { color: #6ea8fe; font-size: 24px; font-weight: 700; letter-spacing: .24em; text-transform: uppercase; }
h1 { margin: 22px 0 18px; font-size: 70px; line-height: 1.08; letter-spacing: -.02em; }
h2 { margin: 18px 0 14px; font-size: 45px; }
p, li { color: #b9c7dc; font-size: 27px; line-height: 1.5; }
.sub { color: #9aacC7; font-size: 31px; line-height: 1.5; max-width: 1420px; }
.rule { width: 88px; height: 4px; background: #6ea8fe; margin: 26px 0; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-top: 34px; }
.grid.two { grid-template-columns: repeat(2, 1fr); }
.panel { background: #151f33; border: 1px solid #2c3c59; border-radius: 15px; padding: 25px 28px; }
.value { color: #fff; font-size: 42px; font-weight: 700; margin-top: 8px; }
.label { color: #91a4c2; font-size: 21px; text-transform: uppercase; letter-spacing: .12em; }
pre { background: #111a2b; border: 1px solid #2c3c59; border-radius: 14px; color: #dce7f8;
      font: 23px/1.45 Consolas, monospace; padding: 25px; white-space: pre-wrap; overflow: hidden; }
.flow { display:flex; align-items:center; justify-content:center; gap:15px; margin-top:70px; flex-wrap:wrap; }
.node { background:#17243a; border:1px solid #35517c; border-radius:12px; padding:20px 24px;
        font-size:25px; font-weight:600; }
.arrow { color:#6ea8fe; font-size:32px; }
.plots { display:grid; grid-template-columns:1fr 1fr; gap:26px; height:700px; margin-top:20px; }
.plots img { width:100%; height:100%; object-fit:contain; background:white; border-radius:14px; }
.prediction { display:grid; grid-template-columns:360px 1fr; gap:30px; align-items:center; }
.prediction img { width:360px; height:300px; object-fit:cover; border-radius:16px; }
.ok { color:#7ee2a8; } .warn { color:#ffd166; }
</style>
"""


def mark(title: str) -> None:
    TIMELINE.append((time.monotonic() - STARTED, title))
    print(f"  {title}", flush=True)


def set_html(page: Page, body: str) -> None:
    page.set_content(f"<!doctype html><html><head>{THEME}</head><body>{body}</body></html>")


def card(page: Page, number: str, title: str, subtitle: str, seconds: float) -> None:
    mark(f"{number} - {title}")
    set_html(
        page,
        f"<div class='wrap' style='display:flex;align-items:center'><div>"
        f"<div class='eyebrow'>{html.escape(number)}</div><div class='rule'></div>"
        f"<h1>{html.escape(title)}</h1><div class='sub'>{html.escape(subtitle)}</div>"
        "</div></div>",
    )
    page.wait_for_timeout(int(seconds * 1000))


def caption(page: Page, title: str, subtitle: str = "") -> None:
    page.evaluate(
        """([title, subtitle]) => {
          document.getElementById('__demo_caption__')?.remove();
          const el = document.createElement('div'); el.id='__demo_caption__';
          el.style.cssText='position:fixed;left:28px;right:28px;bottom:24px;z-index:2147483647;'
            +'background:rgba(13,19,33,.94);border:1px solid #3b5378;border-radius:12px;'
            +'padding:17px 24px;color:white;font-family:Segoe UI,Arial;box-shadow:0 8px 30px #0008';
          el.innerHTML='<div style="font-size:26px;font-weight:700">'+title+'</div>'+
            '<div style="font-size:20px;color:#b9c7dc;margin-top:4px">'+subtitle+'</div>';
          document.body.appendChild(el);
        }""",
        [title, subtitle],
    )


def visit(page: Page, url: str, title: str, subtitle: str, seconds: float, scroll: int = 0) -> None:
    mark(title)
    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    page.wait_for_timeout(1600)
    if scroll:
        page.mouse.wheel(0, scroll)
        page.wait_for_timeout(900)
    caption(page, title, subtitle)
    page.wait_for_timeout(int(seconds * 1000))


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{path.suffix.lstrip('.')};base64,{encoded}"


def json_block(value: object) -> str:
    return html.escape(json.dumps(value, indent=2, sort_keys=True))


def latest_run(workflow: str, conclusion: str | None = None) -> dict[str, str]:
    command = [
        "gh", "run", "list", "--repo", "Aastha-Sakshi/mlopsassinment3",
        "--workflow", workflow, "--limit", "10", "--json",
        "databaseId,status,conclusion,url,headSha",
    ]
    runs = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    if conclusion:
        runs = [run for run in runs if run["conclusion"] == conclusion]
    if not runs:
        raise RuntimeError(f"No {conclusion or 'matching'} {workflow} run found")
    return runs[0]


def live_evidence() -> dict[str, object]:
    health = requests.get(f"{API}/health", timeout=15)
    health.raise_for_status()
    samples = {
        "Known cat": ROOT / "data/processed/PetImages224/test/cat/10.jpg",
        "Known dog": ROOT / "data/processed/PetImages224/test/dog/10060.jpg",
    }
    predictions = {}
    for name, path in samples.items():
        with path.open("rb") as handle:
            response = requests.post(
                f"{API}/predict",
                files={"file": (path.name, handle, "image/jpeg")},
                timeout=30,
            )
        response.raise_for_status()
        predictions[name] = {"path": path, "response": response.json()}
    invalid = requests.post(
        f"{API}/predict",
        files={"file": ("invalid.txt", b"not an image", "text/plain")},
        timeout=30,
    )
    if predictions["Known cat"]["response"]["label"] != "cat":
        raise RuntimeError("Known cat demo image was not classified as cat")
    if predictions["Known dog"]["response"]["label"] != "dog":
        raise RuntimeError("Known dog demo image was not classified as dog")
    if invalid.status_code != 400:
        raise RuntimeError(f"Expected invalid-image status 400, found {invalid.status_code}")
    metrics = requests.get(f"{API}/metrics", timeout=15)
    metrics.raise_for_status()
    return {
        "health": health.json(),
        "predictions": predictions,
        "invalid": {"status_code": invalid.status_code, "response": invalid.json()},
        "metrics": metrics.json(),
    }


def record(page: Page, ci: dict[str, str], cd: dict[str, str], evidence: dict[str, object]) -> None:
    card(page, "CATS VS DOGS", "From images to a deployed service",
         "MLOps Assignment 2 · reproducible data, tracked experiments, API, CI/CD and monitoring", 8)

    mark("01 - End-to-end workflow")
    set_html(page, "<div class='wrap'><div class='eyebrow'>01 · Project</div><h1>One model, complete workflow</h1>"
             "<div class='flow'><div class='node'>Kaggle</div><div class='arrow'>→</div>"
             "<div class='node'>DVC</div><div class='arrow'>→</div><div class='node'>SimpleCNN</div>"
             "<div class='arrow'>→</div><div class='node'>MLflow</div><div class='arrow'>→</div>"
             "<div class='node'>FastAPI</div><div class='arrow'>→</div><div class='node'>Docker</div>"
             "<div class='arrow'>→</div><div class='node'>GitHub Actions</div></div>"
             "<p style='margin-top:70px'>The repository holds the code and configuration. Kaggle supplied compute; the deployed service uses only the finished model.</p></div>")
    page.wait_for_timeout(14_000)

    metadata = json.loads((ROOT / "data/manifests/baseline_50/subset_metadata.json").read_text())
    mark("02 - Dataset and DVC")
    set_html(page, "<div class='wrap'><div class='eyebrow'>02 · M1</div><h1>Versioned data, reproducible subset</h1>"
             "<div class='grid'><div class='panel'><div class='label'>Valid processed images</div><div class='value'>24,966</div></div>"
             "<div class='panel'><div class='label'>Baseline population</div><div class='value'>12,480</div></div>"
             "<div class='panel'><div class='label'>Class balance</div><div class='value'>50 / 50</div></div></div>"
             "<div class='grid two'><div class='panel'><div class='label'>Split</div><pre>Train       9,984\nValidation  1,248\nTest        1,248</pre></div>"
             f"<div class='panel'><div class='label'>Traceability</div><pre>Algorithm: {metadata['algorithm_version']}\n"
             f"Selection seed: {metadata['selection_seed']}\nSubset hash: {metadata['subset_manifest_combined_sha256'][:24]}…\n"
             f"Processed DVC: {metadata['source_processed_dvc']['hash']}</pre></div></div>"
             "<p>The full Kaggle dataset remains the authoritative DVC-versioned source. The subset is represented only by Git-tracked manifests.</p></div>")
    page.wait_for_timeout(22_000)

    visit(page, f"{MLFLOW}/#/experiments/1/runs/{BASELINE_RUN_ID}",
          "03 - MLflow experiment", "Parameters, metrics and saved artifacts from the Kaggle T4 baseline run", 21)
    mark("03 - Evaluation artifacts")
    confusion = image_uri(ROOT / "artifacts/training/simple-cnn-baseline-50pct/confusion_matrix.png")
    loss = image_uri(ROOT / "artifacts/training/simple-cnn-baseline-50pct/loss_curve.png")
    set_html(page, f"<div class='wrap'><div class='eyebrow'>03 · M1</div><h2>Measured baseline results · 80.05% accuracy · 79.99% macro F1</h2>"
             f"<div class='plots'><img src='{confusion}' alt='Confusion matrix'><img src='{loss}' alt='Loss curve'></div></div>")
    page.wait_for_timeout(14_000)

    visit(page, REPOSITORY, "04 - GitHub repository", "Readable project overview, source, manifests, tests and deployment files", 17, 380)
    visit(page, ci["url"], "05 - Continuous integration",
          f"Successful run {ci['databaseId']}: tests, model validation, Docker build and GHCR publish", 17)
    visit(page, PACKAGE, "06 - GHCR image", f"Container image published from commit {ci['headSha'][:12]}", 13)
    visit(page, cd["url"], "07 - Continuous deployment",
          f"Successful run {cd['databaseId']}: immutable image pulled, Compose deployed, smoke test passed", 16)

    visit(page, f"{API}/docs", "08 - FastAPI service", "Health, image prediction and metrics endpoints", 17, 500)

    for number, name in (("09A", "Known cat"), ("09B", "Known dog")):
        item = evidence["predictions"][name]
        mark(f"{number} - {name} prediction")
        set_html(page, f"<div class='wrap'><div class='eyebrow'>{number} · Live prediction</div><h1>{name}</h1>"
                 f"<div class='prediction'><img src='{image_uri(item['path'])}'><pre>{json_block(item['response'])}</pre></div>"
                 "<p class='ok'>The deployed Docker service returned the expected label.</p></div>")
        page.wait_for_timeout(12_000)

    mark("09C - Invalid input handling")
    set_html(page, f"<div class='wrap'><div class='eyebrow'>09C · API validation</div><h1>Invalid input is rejected</h1>"
             f"<div class='grid two'><div class='panel'><div class='label'>HTTP status</div><div class='value warn'>{evidence['invalid']['status_code']}</div></div>"
             f"<div class='panel'><div class='label'>Response</div><pre>{json_block(evidence['invalid']['response'])}</pre></div></div>"
             "<p>The service reports a client error instead of attempting inference on unreadable data.</p></div>")
    page.wait_for_timeout(10_000)

    logs = subprocess.run(
        ["docker", "compose", "logs", "--tail", "10", "api"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    mark("10 - Monitoring")
    metrics = evidence["metrics"]
    set_html(page, "<div class='wrap'><div class='eyebrow'>10 · M5</div><h1>Requests and latency are visible</h1>"
             f"<div class='grid'><div class='panel'><div class='label'>Prediction requests</div><div class='value'>{metrics['prediction_request_count']}</div></div>"
             f"<div class='panel'><div class='label'>Errors</div><div class='value'>{metrics['prediction_error_count']}</div></div>"
             f"<div class='panel'><div class='label'>Average latency</div><div class='value'>{metrics['average_prediction_latency_ms']:.1f} ms</div></div></div>"
             f"<pre style='font-size:18px;margin-top:28px'>{html.escape(logs[-1800:])}</pre></div>")
    page.wait_for_timeout(17_000)

    post = json.loads((ROOT / "artifacts/post_deployment/evaluation.json").read_text())
    mark("11 - Post-deployment evaluation")
    set_html(page, "<div class='wrap'><div class='eyebrow'>11 · M5</div><h1>Labelled batch after deployment</h1>"
             f"<div class='grid'><div class='panel'><div class='label'>Samples</div><div class='value'>{post['sample_count']}</div></div>"
             f"<div class='panel'><div class='label'>Balanced</div><div class='value'>10 + 10</div></div>"
             f"<div class='panel'><div class='label'>Accuracy</div><div class='value'>{post['accuracy'] * 100:.0f}%</div></div></div>"
             f"<pre style='margin-top:34px'>{json_block(post['confusion'])}</pre>"
             "<p>This small check verifies the deployed API with known labels; it is separate from the held-out test evaluation.</p></div>")
    page.wait_for_timeout(15_000)

    elapsed = time.monotonic() - STARTED
    remaining = max(8.0, TARGET_SECONDS - elapsed)
    card(page, "COMPLETE", "M1–M5 demonstrated",
         "DVC data · MLflow model · FastAPI and Docker · tested CI/CD · monitoring and evaluation", remaining)


def convert(webm: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(webm),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:1080", "-movflags", "+faststart", "-an", str(output),
        ],
        check=True,
    )


def write_timeline(path: Path) -> None:
    lines = [f"{int(at) // 60}:{int(at) % 60:02d}  {title}" for at, title in TIMELINE]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight(ci: dict[str, str], cd: dict[str, str], evidence: dict[str, object]) -> None:
    output = ROOT / "artifacts/demo_preflight"
    output.mkdir(parents=True, exist_ok=True)
    pages = [
        ("01_repository.png", REPOSITORY),
        ("02_ci.png", ci["url"]),
        ("03_package.png", PACKAGE),
        ("04_cd.png", cd["url"]),
        ("05_mlflow.png", f"{MLFLOW}/#/experiments/1/runs/{BASELINE_RUN_ID}"),
        ("06_fastapi.png", f"{API}/docs"),
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        for name, url in pages:
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(6_000 if "mlflow" in name else 1_800)
            page.screenshot(path=str(output / name))
            print(f"Preflight screenshot: {name}")
        browser.close()
    print(json.dumps({"health": evidence["health"], "metrics": evidence["metrics"]}, indent=2))


def main() -> None:
    global STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/demo.mp4")
    parser.add_argument("--allow-pending-cd", action="store_true", help="preview only")
    parser.add_argument("--preflight", action="store_true", help="check pages and write screenshots only")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("FFmpeg and FFprobe must be available on PATH")
    for url in (f"{API}/health", MLFLOW):
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    ci = latest_run("CI", "success")
    try:
        cd = latest_run("CD", "success")
    except RuntimeError:
        if not args.allow_pending_cd:
            raise SystemExit("No successful CD run exists; final recording is blocked")
        cd = latest_run("CD")
    evidence = live_evidence()
    if args.preflight:
        preflight(ci, cd, evidence)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="cats_dogs_demo_"))
    print(f"Recording staging directory: {staging}")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=str(staging),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        page = context.new_page()
        page.set_default_timeout(30_000)
        STARTED = time.monotonic()
        record(page, ci, cd, evidence)
        video = page.video
        context.close()
        webm = Path(video.path())
        browser.close()
    print(f"WebM completed: {webm} ({webm.stat().st_size / 1e6:.1f} MB)")
    print("FFmpeg conversion started", flush=True)
    convert(webm, args.output)
    write_timeline(args.output.with_name(args.output.stem + "_timeline.txt"))
    from scripts.validate_video import validate

    result = validate(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
