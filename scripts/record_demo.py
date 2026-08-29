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
body { background: #0d1321; color: #e7edf8; font-family: 'Segoe UI', Arial, sans-serif;
       background-image: radial-gradient(circle at 78% 18%, #17335a66 0, transparent 34%),
                         linear-gradient(135deg, #0b1120 0%, #101a2c 100%); }
body::before { content:''; position:fixed; inset:0; pointer-events:none; opacity:.22;
  background-image:linear-gradient(#6682a918 1px,transparent 1px),linear-gradient(90deg,#6682a918 1px,transparent 1px);
  background-size:54px 54px; }
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
.title-card { position:relative; display:flex; flex-direction:column; justify-content:center; }
.title-card h1 { max-width:1420px; font-size:82px; margin:24px 0 18px; }
.title-card .lead { color:#b8c8de; font-size:33px; max-width:1280px; line-height:1.45; }
.title-rule { width:112px; height:4px; border-radius:4px; background:linear-gradient(90deg,#6ea8fe,#7ee2d6); }
.tech-row { display:flex; gap:13px; margin:32px 0 38px; }
.tech { padding:10px 17px; border:1px solid #36547c; border-radius:999px; color:#cfe0f8;
        background:#14223aaa; font-size:19px; }
.meta-row { display:grid; grid-template-columns:1.1fr .85fr 1.3fr 1.5fr; gap:24px; max-width:1510px; }
.meta-item { border-top:1px solid #334865; padding-top:15px; }
.meta-value { color:#f3f7fd; font-size:24px; margin-top:7px; }
.orb { position:absolute; right:125px; top:135px; width:255px; height:255px; border-radius:50%;
       border:1px solid #6ea8fe55; box-shadow:0 0 90px #397bd844 inset,0 0 100px #397bd822; }
.orb::before,.orb::after { content:''; position:absolute; border-radius:50%; border:1px solid #7ee2d655; }
.orb::before { inset:37px; } .orb::after { inset:82px; background:#6ea8fe22; }
.reveal { animation:rise .75s cubic-bezier(.2,.8,.2,1) both; }
.delay-1 { animation-delay:.15s; } .delay-2 { animation-delay:.3s; } .delay-3 { animation-delay:.45s; }
.orb { animation:breathe 5s ease-in-out infinite; }
@keyframes rise { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:none; } }
@keyframes breathe { 0%,100% { transform:scale(1); opacity:.8; } 50% { transform:scale(1.035); opacity:1; } }
</style>
"""


def mark(title: str) -> None:
    TIMELINE.append((time.monotonic() - STARTED, title))
    print(f"  {title}", flush=True)


def set_html(page: Page, body: str) -> None:
    page.set_content(f"<!doctype html><html><head>{THEME}</head><body>{body}</body></html>")


def hold(page: Page, seconds: float, scroll: int = 0) -> None:
    """Keep a scene alive with deterministic, understated motion."""
    points = ((1510, 180), (1640, 310), (1420, 420), (1580, 520), (1470, 610))
    steps = max(1, min(len(points), int(seconds // 2.2)))
    scroll_step = int(scroll / steps) if scroll else 0
    pause_ms = max(450, int(seconds * 1000 / steps))
    for index in range(steps):
        x, y = points[index]
        page.evaluate(
            """([x, y, duration]) => {
              let cursor = document.getElementById('__demo_cursor__');
              if (!cursor) {
                cursor = document.createElement('div'); cursor.id='__demo_cursor__';
                cursor.style.cssText='position:fixed;left:0;top:0;width:14px;height:14px;border-radius:50%;'
                  +'border:2px solid rgba(255,255,255,.9);background:rgba(110,168,254,.65);'
                  +'box-shadow:0 2px 10px rgba(0,0,0,.45);z-index:2147483647;pointer-events:none;'
                  +'transform:translate(1500px,180px);'; document.body.appendChild(cursor);
              }
              cursor.animate(
                [{transform:cursor.style.transform},{transform:`translate(${x}px,${y}px)`}],
                {duration,fill:'forwards',easing:'cubic-bezier(.22,.75,.28,1)'}
              );
              cursor.style.transform=`translate(${x}px,${y}px)`;
            }""",
            [x, y, min(1800, pause_ms - 120)],
        )
        if scroll_step:
            page.mouse.wheel(0, scroll_step)
        page.wait_for_timeout(pause_ms)


def title_card(page: Page, seconds: float) -> None:
    mark("MLOps Assignment 2 - Cats vs Dogs")
    set_html(
        page,
        "<div class='wrap title-card'><div class='orb'></div>"
        "<div class='eyebrow reveal'>AIMLCZG523 · MLOps · Assignment 2</div>"
        "<div class='title-rule reveal delay-1'></div>"
        "<h1 class='reveal delay-1'>Cats vs Dogs<br>MLOps Pipeline</h1>"
        "<div class='lead reveal delay-2'>From versioned image data and experiment tracking to a tested, "
        "containerized and monitored prediction service.</div>"
        "<div class='tech-row reveal delay-2'><span class='tech'>DVC</span><span class='tech'>SimpleCNN</span>"
        "<span class='tech'>MLflow</span><span class='tech'>FastAPI</span><span class='tech'>Docker</span>"
        "<span class='tech'>GitHub Actions</span></div>"
        "<div class='meta-row reveal delay-3'>"
        "<div class='meta-item'><div class='label'>Submitted by</div><div class='meta-value'>Aastha Sakshi</div></div>"
        "<div class='meta-item'><div class='label'>BITS ID</div><div class='meta-value'>2024AC05266</div></div>"
        "<div class='meta-item'><div class='label'>Use case</div><div class='meta-value'>Pet adoption · Binary vision</div></div>"
        "<div class='meta-item'><div class='label'>Coverage</div><div class='meta-value'>M1–M5 · End-to-end workflow</div></div>"
        "</div></div>",
    )
    hold(page, seconds)


def card(page: Page, number: str, title: str, subtitle: str, seconds: float) -> None:
    mark(f"{number} - {title}")
    set_html(
        page,
        f"<div class='wrap' style='display:flex;align-items:center'><div>"
        f"<div class='eyebrow'>{html.escape(number)}</div><div class='rule'></div>"
        f"<h1>{html.escape(title)}</h1><div class='sub'>{html.escape(subtitle)}</div>"
        "</div></div>",
    )
    hold(page, seconds)


def caption(page: Page, title: str, subtitle: str = "") -> None:
    page.evaluate(
        """([title, subtitle]) => {
          document.getElementById('__demo_caption__')?.remove();
          const el = document.createElement('div'); el.id='__demo_caption__';
          el.style.cssText='position:fixed;right:28px;top:24px;width:620px;z-index:2147483646;'
            +'background:rgba(13,19,33,.94);border:1px solid #3b5378;border-radius:12px;'
            +'padding:13px 18px;color:white;font-family:Segoe UI,Arial;box-shadow:0 8px 30px #0006';
          el.innerHTML='<div style="font-size:21px;font-weight:700">'+title+'</div>'+
            '<div style="font-size:16px;color:#b9c7dc;margin-top:3px">'+subtitle+'</div>';
          document.body.appendChild(el);
        }""",
        [title, subtitle],
    )


def visit(page: Page, url: str, title: str, subtitle: str, seconds: float, scroll: int = 0) -> None:
    mark(title)
    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    if url.startswith(REPOSITORY) and "#readme" in url:
        page.evaluate("window.scrollTo(0, 850)")
        scroll = max(0, scroll - 850)
    page.wait_for_timeout(1800)
    caption(page, title, subtitle)
    hold(page, seconds, scroll=scroll)


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
    title_card(page, 10)

    mark("01 - End-to-end workflow")
    set_html(page, "<div class='wrap'><div class='eyebrow'>01 · Project</div><h1>One model, complete workflow</h1>"
             "<div class='flow'><div class='node'>Kaggle</div><div class='arrow'>→</div>"
             "<div class='node'>DVC</div><div class='arrow'>→</div><div class='node'>SimpleCNN</div>"
             "<div class='arrow'>→</div><div class='node'>MLflow</div><div class='arrow'>→</div>"
             "<div class='node'>FastAPI</div><div class='arrow'>→</div><div class='node'>Docker</div>"
             "<div class='arrow'>→</div><div class='node'>GitHub Actions</div></div>"
             "<p style='margin-top:70px'>The repository holds the code and configuration. Kaggle supplied compute; the deployed service uses only the finished model.</p></div>")
    hold(page, 14)

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
    hold(page, 22)

    visit(page, f"{MLFLOW}/#/experiments/1/runs/{BASELINE_RUN_ID}",
          "03 - MLflow experiment", "Parameters, metrics and saved artifacts from the Kaggle T4 baseline run", 21)
    mark("03 - Evaluation artifacts")
    confusion = image_uri(ROOT / "artifacts/training/simple-cnn-baseline-50pct/confusion_matrix.png")
    loss = image_uri(ROOT / "artifacts/training/simple-cnn-baseline-50pct/loss_curve.png")
    set_html(page, f"<div class='wrap'><div class='eyebrow'>03 · M1</div><h2>Measured baseline results · 80.05% accuracy · 79.99% macro F1</h2>"
             f"<div class='plots'><img src='{confusion}' alt='Confusion matrix'><img src='{loss}' alt='Loss curve'></div></div>")
    hold(page, 14)

    visit(page, f"{REPOSITORY}#readme", "04 - GitHub repository", "Readable project overview, source, manifests, tests and deployment files", 17, 1050)
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
        hold(page, 12)

    mark("09C - Invalid input handling")
    set_html(page, f"<div class='wrap'><div class='eyebrow'>09C · API validation</div><h1>Invalid input is rejected</h1>"
             f"<div class='grid two'><div class='panel'><div class='label'>HTTP status</div><div class='value warn'>{evidence['invalid']['status_code']}</div></div>"
             f"<div class='panel'><div class='label'>Response</div><pre>{json_block(evidence['invalid']['response'])}</pre></div></div>"
             "<p>The service reports a client error instead of attempting inference on unreadable data.</p></div>")
    hold(page, 10)

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
    hold(page, 17)

    post = json.loads((ROOT / "artifacts/post_deployment/evaluation.json").read_text())
    mark("11 - Post-deployment evaluation")
    set_html(page, "<div class='wrap'><div class='eyebrow'>11 · M5</div><h1>Labelled batch after deployment</h1>"
             f"<div class='grid'><div class='panel'><div class='label'>Samples</div><div class='value'>{post['sample_count']}</div></div>"
             f"<div class='panel'><div class='label'>Balanced</div><div class='value'>10 + 10</div></div>"
             f"<div class='panel'><div class='label'>Accuracy</div><div class='value'>{post['accuracy'] * 100:.0f}%</div></div></div>"
             f"<pre style='margin-top:34px'>{json_block(post['confusion'])}</pre>"
             "<p>This small check verifies the deployed API with known labels; it is separate from the held-out test evaluation.</p></div>")
    hold(page, 15)

    elapsed = time.monotonic() - STARTED
    remaining = max(8.0, TARGET_SECONDS - elapsed)
    card(page, "COMPLETE", "M1–M5 demonstrated",
         "DVC data · MLflow model · FastAPI and Docker · tested CI/CD · monitoring and evaluation", remaining)


def convert(webm: Path, output: Path, subtitles: Path) -> None:
    subtitle_path = subtitles.relative_to(ROOT).as_posix()
    video_filter = (
        "scale=1920:1080,"
        f"subtitles=filename='{subtitle_path}':force_style='FontName=Segoe UI,FontSize=18,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&HCC0A1020,BackColour=&H990A1020,"
        "BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginV=34'"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(webm),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-vf", video_filter, "-movflags", "+faststart", "-an", str(output),
        ],
        check=True, cwd=ROOT,
    )


def write_timeline(path: Path) -> None:
    lines = [f"{int(at) // 60}:{int(at) % 60:02d}  {title}" for at, title in TIMELINE]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def srt_time(seconds: float) -> str:
    milliseconds = max(0, int(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_subtitles(path: Path) -> None:
    narration = {
        "MLOps Assignment 2 - Cats vs Dogs": "This project takes a cats-versus-dogs model from image data to a working, monitored API.",
        "01 - End-to-end workflow": "The same project code connects data preparation, training, testing, deployment and monitoring.",
        "02 - Dataset and DVC": "The full dataset stays versioned with DVC. I use a balanced, repeatable subset for the baseline run.",
        "03 - MLflow experiment": "Here is the completed MLflow run, with the settings and results from training on Kaggle.",
        "03 - Evaluation artifacts": "On the held-out test set, the baseline reached 80.05 percent accuracy and 79.99 percent macro F1.",
        "04 - GitHub repository": "The repository keeps the code, configuration, data lists, tests and deployment files in one place.",
        "05 - Continuous integration": "A push to main runs the tests, checks the saved model, builds Docker and publishes the image.",
        "06 - GHCR image": "This is the Docker image produced by the successful workflow and stored in GitHub's registry.",
        "07 - Continuous deployment": "The Windows runner pulled that image, updated the service and checked a real prediction.",
        "08 - FastAPI service": "The deployed API provides a health check, image prediction and simple request metrics.",
        "09A - Known cat prediction": "This known cat image is classified correctly, with probabilities for both classes.",
        "09B - Known dog prediction": "The same deployed service also returns the expected result for this known dog image.",
        "09C - Invalid input handling": "If the upload is not a valid image, the API returns a clear error instead of trying to predict.",
        "10 - Monitoring": "The service records how many predictions were made, any errors, and the average response time.",
        "11 - Post-deployment evaluation": "Finally, I checked the live API with 20 labelled images: 10 cats and 10 dogs.",
        "COMPLETE - M1–M5 demonstrated": "That completes the full assignment path, from versioned data to a deployed and checked model.",
    }
    blocks = []
    for index, (start, title) in enumerate(TIMELINE, start=1):
        # Keep the closing caption inside the recorded card even when browser
        # shutdown trims a small amount from the target duration.
        next_start = TIMELINE[index][0] if index < len(TIMELINE) else start + 7.4
        end = max(start + 2.5, next_start - 0.6)
        text = narration.get(title, title)
        blocks.append(f"{index}\n{srt_time(start + 0.6)} --> {srt_time(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def preflight(ci: dict[str, str], cd: dict[str, str], evidence: dict[str, object]) -> None:
    output = ROOT / "artifacts/demo_preflight"
    output.mkdir(parents=True, exist_ok=True)
    pages = [
        ("01_repository.png", f"{REPOSITORY}#readme"),
        ("02_ci.png", ci["url"]),
        ("03_package.png", PACKAGE),
        ("04_cd.png", cd["url"]),
        ("05_mlflow.png", f"{MLFLOW}/#/experiments/1/runs/{BASELINE_RUN_ID}"),
        ("06_fastapi.png", f"{API}/docs"),
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        title_card(page, 0.2)
        page.screenshot(path=str(output / "00_title.png"))
        print("Preflight screenshot: 00_title.png")
        for name, url in pages:
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(6_000 if "mlflow" in name else 1_800)
            if "repository" in name:
                page.mouse.wheel(0, 1050)
                page.wait_for_timeout(1_000)
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
    timeline_path = args.output.with_name(args.output.stem + "_timeline.txt")
    subtitle_path = args.output.with_name(args.output.stem + "_subtitles.srt")
    write_timeline(timeline_path)
    write_subtitles(subtitle_path)
    print("FFmpeg conversion started", flush=True)
    convert(webm, args.output, subtitle_path)
    from scripts.validate_video import validate

    result = validate(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
