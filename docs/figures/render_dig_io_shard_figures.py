#!/usr/bin/env python3
"""Render dig_io summary tables to SVG + PNG (Korean, Noto Sans CJK).

Requires: matplotlib, system font Noto Sans CJK (e.g. fonts-noto-cjk on Debian).

Usage (from repo root):
  .venv/bin/python docs/figures/render_dig_io_shard_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table


HERE = Path(__file__).resolve().parent
OUT_SVG = HERE / "dig_io_shard_figure.svg"
OUT_PNG = HERE / "dig_io_shard_figure.png"

# Prefer installed Noto CJK (matplotlib may register TTC as JP; glyphs cover Korean).
_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _font() -> fm.FontProperties:
    if os.path.isfile(_NOTO):
        return fm.FontProperties(fname=_NOTO)
    print(
        "WARNING: Noto Sans CJK not found at",
        _NOTO,
        "— Korean glyphs may be missing. Install fonts-noto-cjk (Debian/Ubuntu).",
    )
    return fm.FontProperties()


def _style_table(tbl: Table, font: fm.FontProperties, header_color: str = "#e8e8e8") -> None:
    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(fontproperties=font, fontsize=8.5)
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 == 1 else "#f7f7f7")


def main() -> None:
    font = _font()
    plt.rcParams["axes.unicode_minus"] = False
    if os.path.isfile(_NOTO):
        name = font.get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"

    # --- Table data (line breaks tuned for ~14" width) ---
    fig1_cols = ["블록", "역할", "Python에서 쓰는 곳(예)"]
    fig1_rows = [
        [
            "대조 뷰(view) 커널",
            "edge_index 등 입력, 결정적(seed)\n서브그래프 샘플링 등 CPU 처리,\nnumpy 반환",
            "GraphCL 등 view 구현의\nRust 경로",
        ],
        [
            "스캐폴드 분할",
            "scaffold_bucket_split /\nscaffold_bucket_sort\n(버킷 경계·정렬)",
            "dig/.../dataset.py\nkey_split(..., impl='rust')",
        ],
        [
            "Molecule shard",
            "MoleculeShardWriter /\nMoleculeShardReader\n(PyG Data 동등 패킹)",
            "MoleculeNetShard.py\nconvert_dataset_to_shard.py",
        ],
    ]

    fig2_cols = ["구분", "QDF (qdf_io)", "DGCL (dig_io)"]
    fig2_rows = [
        ["매직", "QDFSHRD\\0", "DIGSHRD\\0"],
        [
            "담는 것",
            "전처리 1분자: 격자·거리행렬·\n포텐셜 등 3D 필드 레코드",
            "MoleculeNet PyG Data 1개\n(z, pos, edge_index …,\n선택 MMFF 4슬롯·에너지)",
        ],
        [
            "옆 파일",
            "train_*_shard.bin 등\n(전처리 트리)",
            "{root}/{name}/processed/\ndata.pt 옆 data.shard",
        ],
        [
            "읽기",
            "ShardReader + np.load 등\n이어 읽기",
            "mmap + 인덱스 테이블,\n레코드 단위 디코드",
        ],
        [
            "목적",
            "대량 .npy IO 완화",
            "data.pt 전체 상주 부담 완화,\n로더 I/O 개선",
        ],
    ]

    fig3_cols = ["항목", "경로"]
    fig3_rows = [
        ["Rust·포맷 주석", "dig_io/src/lib.rs"],
        ["Python 바인딩", "dig_io/python/dig_io/__init__.py"],
        ["Shard 데이터셋", "dig/threedgraph/dataset/MoleculeNetShard.py"],
        ["스캐폴드 Rust 위임", "dig/threedgraph/dataset/dataset.py\n(key_split, _try_dig_io)"],
        ["변환 스크립트", "examples/sslgraph/convert_dataset_to_shard.py"],
        ["파이프라인 문서", "docs/qdf_to_dgcl_pipeline.md\ndocs/qdf_to_dgcl_pipeline_two_tier.md"],
    ]

    fig = plt.figure(figsize=(14, 18), dpi=120)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "DGCL dig_io 요약 도표",
        fontproperties=font,
        fontsize=15,
        y=0.98,
    )

    gs = GridSpec(3, 1, figure=fig, left=0.06, right=0.94, top=0.93, bottom=0.04, hspace=0.28)

    specs = [
        ("Fig. 1  dig_io 세 블록", fig1_cols, fig1_rows, [0.14, 0.38, 0.48]),
        ("Fig. 2  QDF shard vs DGCL shard", fig2_cols, fig2_rows, [0.12, 0.44, 0.44]),
        ("Fig. 3  관련 경로", fig3_cols, fig3_rows, [0.22, 0.78]),
    ]

    for i, (title, cols, rows, widths) in enumerate(specs):
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        ax.set_title(title, fontproperties=font, fontsize=12, loc="left", pad=8)

        tbl = ax.table(
            cellText=rows,
            colLabels=cols,
            loc="upper center",
            cellLoc="left",
            colWidths=widths,
            bbox=[0, 0.02, 1, 0.92],
        )
        tbl.auto_set_font_size(False)
        tbl.scale(1, 2.1)
        _style_table(tbl, font)

    fig.savefig(OUT_SVG, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PNG, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_SVG}\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
