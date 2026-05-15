#!/usr/bin/env python3
"""Extract ```mermaid``` blocks from docs/*.md and render to SVG (+ PNG when possible).

**SVG** (default): POST to `kroki.io` (needs network). Uses a browser-like User-Agent
because Kroki/Cloudflare may return HTTP 403 to Python's default urllib UA.

**PNG**: (1) Kroki ``/mermaid/png`` — large diagrams sometimes return HTTP 400 from the
server; (2) if ``rsvg-convert`` is on PATH, rasterize the SVG we just wrote instead.

**Local** (offline): ``--renderer mmdc`` uses ``mmdc`` on PATH, or
``npx --yes @mermaid-js/mermaid-cli`` (needs Node + working Chromium for Puppeteer).

Usage (repo root):

  .venv/bin/python docs/figures/render_mermaid_diagrams.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
OUT_DIR = DOCS / "figures" / "mermaid"
KROKI_SVG = "https://kroki.io/mermaid/svg"
KROKI_PNG = "https://kroki.io/mermaid/png"

JOBS: list[tuple[Path, list[str]]] = [
    (DOCS / "qdf_to_dgcl_pipeline.md", ["qdf_to_dgcl_pipeline"]),
    (
        DOCS / "qdf_to_dgcl_pipeline_two_tier.md",
        ["qdf_to_dgcl_two_tier_data", "qdf_to_dgcl_two_tier_code"],
    ),
]

_KROKI_HEADERS = {
    "Content-Type": "text/plain; charset=utf-8",
    "User-Agent": "Mozilla/5.0 (compatible; 3DGCL-docs-render/1.0)",
}


def extract_mermaid_blocks(md_text: str) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", md_text, flags=re.DOTALL)


def render_kroki(body: str, url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        method="POST",
        headers=_KROKI_HEADERS,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def mmdc_base_cmd() -> list[str]:
    exe = shutil.which("mmdc")
    if exe:
        return [exe]
    npx = shutil.which("npx")
    if not npx:
        raise RuntimeError("Neither mmdc nor npx on PATH")
    return [npx, "--yes", "@mermaid-js/mermaid-cli"]


def render_mmdc(mmd_path: Path, out_path: Path, puppeteer_config: Path | None) -> None:
    fmt = out_path.suffix.lstrip(".")
    cmd = mmdc_base_cmd() + ["-i", str(mmd_path), "-o", str(out_path), "-b", "transparent"]
    if puppeteer_config and puppeteer_config.is_file():
        cmd.extend(["-p", str(puppeteer_config)])
    if fmt == "png":
        cmd.extend(["-w", "2400", "-H", "1800", "-s", "2"])
    subprocess.run(cmd, check=True)


def try_png_from_svg(svg_path: Path, png_path: Path, width: int = 2800) -> bool:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return False
    subprocess.run(
        [rsvg, "-w", str(width), "-o", str(png_path), str(svg_path)],
        check=True,
        capture_output=True,
    )
    return True


def try_kroki_png(body: str) -> bytes | None:
    try:
        return render_kroki(body, KROKI_PNG, timeout=180)
    except urllib.error.HTTPError as e:
        if e.code in (400, 413, 500, 502, 503):
            return None
        err = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Kroki PNG HTTP {e.code}: {err}") from e
    except urllib.error.URLError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--renderer",
        choices=("kroki", "mmdc"),
        default=os.environ.get("MERMAID_RENDERER", "kroki"),
        help="kroki (default) or local mmdc",
    )
    ap.add_argument(
        "--puppeteer-config",
        type=Path,
        default=DOCS / "figures" / "puppeteer-config.json",
        help="For mmdc: optional Puppeteer JSON (e.g. --no-sandbox)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = OUT_DIR / "src"
    src_dir.mkdir(exist_ok=True)

    for md_path, stems in JOBS:
        if not md_path.is_file():
            print(f"SKIP missing {md_path}", file=sys.stderr)
            continue
        text = md_path.read_text(encoding="utf-8")
        blocks = extract_mermaid_blocks(text)
        if len(blocks) != len(stems):
            print(
                f"ERROR {md_path.name}: expected {len(stems)} mermaid blocks, got {len(blocks)}",
                file=sys.stderr,
            )
            return 1
        for stem, body in zip(stems, blocks):
            body = body.strip() + "\n"
            mmd_path = src_dir / f"{stem}.mmd"
            mmd_path.write_text(body, encoding="utf-8")
            svg_path = OUT_DIR / f"{stem}.svg"
            png_path = OUT_DIR / f"{stem}.png"

            if args.renderer == "kroki":
                try:
                    svg_path.write_bytes(render_kroki(body, KROKI_SVG))
                except urllib.error.HTTPError as e:
                    err = e.read().decode("utf-8", errors="replace")[:2000]
                    print(f"ERROR Kroki SVG {stem}: HTTP {e.code}\n{err}", file=sys.stderr)
                    return 1
                print(f"Wrote {svg_path.relative_to(REPO)}")

                png_data = try_kroki_png(body)
                if png_data:
                    png_path.write_bytes(png_data)
                    print(f"Wrote {png_path.relative_to(REPO)}")
                elif try_png_from_svg(svg_path, png_path):
                    print(f"Wrote {png_path.relative_to(REPO)} (rsvg-convert from SVG)")
                else:
                    print(
                        f"WARN {stem}: Kroki PNG failed and rsvg-convert not available; "
                        f"only {svg_path.name} (install: apt install librsvg2-bin, re-run)",
                        file=sys.stderr,
                    )
            else:
                for ext in ("svg", "png"):
                    out = OUT_DIR / f"{stem}.{ext}"
                    render_mmdc(mmd_path, out, args.puppeteer_config)
                    print(f"Wrote {out.relative_to(REPO)}")

    print("Done. Source .mmd:", src_dir.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
