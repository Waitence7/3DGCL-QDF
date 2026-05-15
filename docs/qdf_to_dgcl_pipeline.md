# QDF → DGCL 연결 프로세스 (3DGCL 레포)

이 문서는 **QuantumDeepField(QDF)** 와 **DGCL(dig/sslgraph, GraphCL 등)** 사이의 데이터·코드 흐름을 한 장에 정리한 도표입니다.  
(채팅에서 공유한 Mermaid와 동일 내용을 파일로 보관합니다.)

## 요약

| 구간 | 내용 |
|------|------|
| **QDF** | `*.txt` → `preprocess.py` + `qdf_io` → `.npy` / `*_shard.bin` → `train.py` → **체크포인트** |
| **브리지** | ESOL 등 **MMFF 슬롯 좌표**에서 **QDF 추론** → CSV → **가중치 `.pt`** (선택: **aux `.pt`**) |
| **DGCL** | MoleculeNet + 대조 뷰 + **(선택) `dig_io` Rust** (Molecule shard mmap, 뷰 커널, 스캐폴드 정렬) → **GraphCL 사전학습** → 파인튜닝/다운스트림 |

> QDF **전처리 산출물**과 DGCL **PyG 그래프**는 포맷이 다릅니다. 실질적인 연결은 **학습된 QDF 체크포인트를 MMFF 좌표 위에서 다시 호출**하는 스크립트 체인입니다.

## 도표 (Mermaid)

GitHub·VS Code 등 Mermaid를 렌더하는 뷰어에서 열면 그림으로 보입니다. **PNG/SVG 정적 산출물**은 [`figures/mermaid/`](figures/mermaid/) 및 [`mermaid/README.md`](figures/mermaid/README.md) 참고.

```mermaid
flowchart TB
  %% ========= QDF (QuantumDeepField_molecule) =========
  subgraph QDF["QuantumDeepField_molecule — QDF 본체"]
    direction TB

    subgraph QDF_IN["입력 / 메타"]
      QM9TXT["QM9 등 dataset/*.txt\n(분자 블록 + (선택) property 줄)"]
      BASIS["basis_set (예: 6-31G)"]
      ORBD["orbital_dict\n(orbitaldict_*.pickle 등)"]
    end

    subgraph QDF_PRE["전처리 파이프라인"]
      PP["train/preprocess.py\npredict/preprocess.py"]
      BE["--backend numpy | rust\n--rust-batch-size"]
      OF["--output-format npy | shard | both"]
      PP --> BE
      PP --> OF
    end

    subgraph QDF_RUST["qdf_io (Rust + PyO3)"]
      PARSE["parse_molecule_block_rust\n(텍스트→배열, AO 라벨)"]
      PB["preprocess_batch_rust / preprocess_molecule_rust\n(필드·거리행렬·potential, Rayon)"]
      SW["ShardWriter\n(*_shard.bin, mmap-friendly)"]
    end

    subgraph QDF_OUT["전처리 산출물"]
      NPY["dir_preprocess/*.npy\n(object array 레코드)"]
      SHARD["train_*_shard.bin 등\n(MyDatasetShard / mmap)"]
    end

    subgraph QDF_TRAIN["QDF 학습·추론 (필드 모델)"]
      DS["MyDataset / MyDatasetShard\n(train/train.py)"]
      QMODEL["QuantumDeepField\n(Energy + HK map / potential 이중 태스크)"]
      CKPT["output / 체크포인트\n(.pkl 등)"]
    end

    subgraph QDF_BENCH["검증·벤치 (선택 경로)"]
      VPRE["verify_preprocess_backend.py"]
      VSH["verify_shard.py"]
      CNV["convert_to_shard.py"]
      PROF["profile_preprocess.py\nprofile_train.py\nprofile_predict.py"]
      SPL["bench/sampler.py\n(run_pipeline.ipynb 등)"]
    end

    QM9TXT --> PP
    BASIS --> PP
    ORBD --> PP
    PP --> PARSE
    PP --> PB
    PP --> SW
    PB --> NPY
    SW --> SHARD
    NPY --> DS
    SHARD --> DS
    DS --> QMODEL --> CKPT

    PP -.-> VPRE
    NPY -.-> CNV
    SHARD -.-> VSH
    PP -.-> PROF
    DS -.-> SPL
  end

  %% ========= Bridge: QDF → DGCL =========
  subgraph BRIDGE["3DGCL 쪽 브리지 — QDF ‘신호’를 DGCL에 주입"]
    direction TB

    subgraph DGCL_DATA["DGCL 입력 데이터 (그래프)"]
      MN["MoleculeNet 계열\n(예: ESOL smiles → PyG Data)"]
      MMFF["MMFF 컨포머 / 슬롯 좌표\n(views_fn, WeightedMMFFView 등)"]
      MN --> MMFF
    end

    subgraph QDF_ON_MMFF["QDF를 ESOL·MMFF 위에서 돌리는 체인"]
      QPRED["examples/sslgraph/bench/qdf_mmff_predict.py\n(슬롯 좌표 + z → QDF record → predict)"]
      CSV["*_qdf_mmff_preds_*.csv\n(homo/lumo 또는 atomization)"]
      CW["examples/sslgraph/bench/compute_mmff_weights.py\n(Boltzmann 또는 QDF 점수 softmax)"]
      WPT["dataset/*_mmff_weights_*.pt\n(+ 메타: qdf_property 등)"]
      AUXB["(선택) build_qdf_aux_ensemble 등\n→ esol_qdf_aux_*.pt"]
    end

    CKPT -->|"체크포인트 로드"| QPRED
    MMFF --> QPRED
    QPRED --> CSV --> CW --> WPT
    CSV -.-> AUXB
  end

  %% ========= DGCL (dig / sslgraph) =========
  subgraph DGCL["dig / sslgraph — DGCL(GraphCL 등)"]
    direction TB

    subgraph DGCL_RUST["dig_io (Rust + PyO3, 선택)"]
      direction TB
      DSH["Molecule shard\nDIGSHRD · processed/data.shard\nWriter / Reader mmap"]
      DV["대조 뷰 커널\nuniform_sample_subgraph 등\n(seed 고정 서브그래프/perturb)"]
      DSP["스캐폴드 분할\nscaffold_bucket_sort\nkey_split(..., impl='rust')"]
    end

    subgraph VIEWS["대조 학습 뷰·샘플링"]
      RV["RandomView / MMFF views"]
      WM["WeightedMMFFView\n(weight_mode=auto →\n가중치 .pt + Boltzmann fallback)"]
    end

    subgraph PRE["사전학습"]
      PNB["examples/sslgraph/pretrain.ipynb\nrun_pretrain_compare.ipynb"]
      QAUX["qdf_aux_io.apply_qdf_aux_from_pt\n(args.qdf_aux_lambda, head)"]
      GCL["GraphCL + SchNet/SphereNet…\n대조 손실 + (선택) QDF 보조 MSE"]
    end

    subgraph POST["다운스트림"]
      FT["finetune.ipynb / downstream.ipynb"]
    end

    DSH -.-> MN
    DV -.-> RV
    DSP -.-> MN
    WPT --> WM
    MMFF --> RV
    MMFF --> WM
    MN --> PNB
    RV --> PNB
    WM --> PNB
    AUXB -.->|"λ>0일 때"| QAUX
    QAUX --> GCL
    PNB --> GCL --> FT
  end

  %% ========= Cross-links =========
  CKPT -.->|"동일 물리 필드 모델 가중치\n(다른 좌표계에서 재사용)"| QPRED
  SHARD -.->|"mmap 단일 파일 shard 아이디어\n(포맷은 QDFSHRD ≠ DIGSHRD)"| DSH

  classDef qdf fill:#e8f4ff,stroke:#3366cc;
  classDef bridge fill:#fff4e6,stroke:#cc6600;
  classDef dgcl fill:#f0ffe8,stroke:#339933;
  class QDF,QDF_IN,QDF_PRE,QDF_RUST,QDF_OUT,QDF_TRAIN,QDF_BENCH qdf;
  class BRIDGE,DGCL_DATA,QDF_ON_MMFF bridge;
  class DGCL,DGCL_RUST,VIEWS,PRE,POST dgcl;
```

### 렌더된 이미지 (Mermaid 미지원 뷰어·인쇄용)

`docs/figures/render_mermaid_diagrams.py`로 재생성합니다.

| PNG | SVG |
|-----|-----|
| [figures/mermaid/qdf_to_dgcl_pipeline.png](figures/mermaid/qdf_to_dgcl_pipeline.png) | [figures/mermaid/qdf_to_dgcl_pipeline.svg](figures/mermaid/qdf_to_dgcl_pipeline.svg) |

![QDF → DGCL 통합 플로차트](figures/mermaid/qdf_to_dgcl_pipeline.png)

## DGCL 쪽 Rust (`dig_io`) — shard·뷰·스캐폴드

QDF의 **`qdf_io`** 가 전처리·`QDFSHRD` shard를 다루는 것과 달리, DGCL용 **`dig_io`** 는 **PyG 그래프 파이프라인**을 위한 선택적 네이티브 확장입니다. 빌드: 레포 루트에서 `cd dig_io && maturin develop --release` 등. Python에서는 `dig_io.is_available()` 이 참일 때만 Rust 경로를 쓰고, 실패 시 순수 Python으로 동작합니다.

### 1) Molecule shard (`DIGSHRD`, `data.shard`)

- **역할:** MoleculeNet이 만든 `Data`와 동등한 텐서·메타를 **한 파일**에 패킹하고 **`mmap`** 으로 `__getitem__` 시점에 읽습니다. `processed/data.pt` 전체를 메모리에 올리는 부담을 줄이거나 I/O 패턴을 단순화할 때 쓰는 **옵션**입니다.
- **위치:** 보통 `{root}/{dataset}/processed/data.pt` 옆의 **`data.shard`**.
- **코드:** `dig_io.MoleculeShardWriter` / `MoleculeShardReader`, Python 쪽 `dig.threedgraph.dataset.MoleculeNetShard`, 변환 `examples/sslgraph/convert_dataset_to_shard.py` (또는 `MoleculeNetShard` 내 변환). 파인튜닝 등에서 **`loader='shard'`** 를 고르면 이 경로를 탑니다.
- **QDF shard와 혼동 주의:** QDF 전처리의 `*_shard.bin` 은 매직 **`QDFSHRD`** 이고 필드 레코드용입니다. DGCL `data.shard` 는 **`DIGSHRD`** 로 **그래프(`z`, `pos`, `edge_index`, …, 선택 MMFF 슬롯)** 용입니다. 이름만 비슷하고 포맷·소비 코드는 완전히 다릅니다.

### 2) 대조 뷰(view) 커널

- **역할:** `edge_index` 등을 넣고 **시드가 고정된** 서브그래프 샘플링·랜덤 워크·엣지 perturb 등을 CPU에서 처리해 numpy로 돌려줍니다. GraphCL 계열에서 뷰 생성 비용을 줄일 때 쓰입니다.
- **연결:** 도표상 `dig_io` 뷰 커널은 **`RandomView` / MMFF 뷰** 쪽 사전학습(`pretrain.ipynb` 등)으로 이어지는 **가속 경로**로 보면 됩니다.

### 3) 스캐폴드 분할 (`scaffold_bucket_sort`)

- **역할:** `dig.threedgraph.dataset.dataset.key_split` 에서 **`impl='rust'`** 일 때, 키(예: 스캐폴드 id) 기준 **안정 정렬·버킷 경계 반올림**을 Rust에 맡깁니다. 동일 RNG 시드면 기존 Torch 구현과 맞추는 것이 목표입니다.
- **연결:** 도표의 `MoleculeNet` 노드는 **데이터셋 구축·split** 단계와 맞물리므로 점선으로 연결했습니다.

### 더 읽기

- 상세 스펙·표: [`dig_io_shard_and_kernels.md`](dig_io_shard_and_kernels.md), 도표만 [`figures/dig_io_shard_figure.png`](figures/dig_io_shard_figure.png) / [`.svg`](figures/dig_io_shard_figure.svg)

## 관련 문서·색인

- 레포 색인: `notebooks_overview.md`
- 연구 노트(구현 디테일): `연구노트.md`

## 파일 위치

- 본 문서: `docs/qdf_to_dgcl_pipeline.md`
