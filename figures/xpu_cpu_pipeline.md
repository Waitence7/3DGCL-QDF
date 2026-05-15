# CPU vs XPU 파이프라인 (3DGCL / QDF 벤치)

이 저장소에서 **Intel XPU(`torch.xpu`)** 와 **CPU** 를 분리해 측정하는 흐름을 정리했습니다. 오케스트레이터는 `examples/sslgraph/bench/ensemble_cpu_xpu_orchestrator.py`, 디바이스 선택은 `dig/sslgraph/utils/device.py` 의 `pick_torch_device()` 입니다.

---

## 1. 디바이스 선택 (`pick_torch_device`) — 일반 학습/평가

`TORCH_DEVICE` 가 비어 있으면 아래 **우선순위**로 첫 매칭 디바이스를 고릅니다. `TORCH_DEVICE=cpu` 또는 `xpu:0` 처럼 지정하면 그대로 사용합니다.

```mermaid
flowchart TD
  START["pick_torch_device()"]
  E1{"TORCH_DEVICE 또는<br/>explicit 인자?"}
  E2{"CUDA 사용 가능?"}
  E3{"Apple MPS?"}
  E4{"torch.xpu 존재 +<br/>TORCH_DISABLE_XPU_DEFAULT<br/>비활성?"}
  E5["XPU: is_available 또는<br/>device_count 또는<br/>버전에 +xpu"]
  E6{"Ascend NPU +<br/>TORCH_SKIP_NPU 아님?"}
  E7{"USE_DIRECTML 또는<br/>TORCH_FALLBACK_DIRECTML?"}
  OUTD["TORCH_DEVICE=directml →<br/>torch_directml.device"]
  OUTE["명시 문자열 →<br/>torch.device(pref)"]
  CUDA["cuda:INDEX"]
  MPS["mps"]
  XPU["xpu:INDEX"]
  NPU["npu:INDEX"]
  DML["DirectML device"]
  CPU["cpu"]

  START --> E1
  E1 -->|예| OUTE
  E1 -->|아니오| E2
  E2 -->|예| CUDA
  E2 -->|아니오| E3
  E3 -->|예| MPS
  E3 -->|아니오| E4
  E4 -->|아니오 / 스킵| E6
  E4 -->|예| E5
  E5 -->|예| XPU
  E5 -->|아니오| E6
  E6 -->|예| NPU
  E6 -->|아니오| E7
  E7 -->|예| DML
  E7 -->|아니오| CPU

  E1 -.->|"pref=directml/dml"| OUTD
```

---

## 2. 앙상블 오케스트레이터 — **동일 파이프라인**을 CPU / XPU 로 각각 1회

`ensemble_cpu_xpu_orchestrator.py` 는 **한 런 디렉터리** 아래에 `cpu/` 와 `xpu/` 를 만들고, 환경만 바꿔 **같은 3단계**를 순차 실행합니다. XPU 런타임이 없으면 XPU 분기는 스킵됩니다.

```mermaid
flowchart TB
  subgraph ORCH["ensemble_cpu_xpu_orchestrator.py"]
    R0["출력:<br/>figs/qdf_ensemble_cpu_xpu_<ts>/"]
    R1{"torch.xpu.is_available()?"}
    R1 -->|아니오| SKIP["xpu 분기 스킵<br/>SUMMARY.md 에 사유 기록"]
    R1 -->|예| BOTH["CPU 모드 실행 후<br/>XPU 모드 실행"]
  end

  subgraph CPU["MODE: cpu 디렉터리"]
    Ecpu["환경:<br/>TORCH_DEVICE=cpu<br/>TORCH_DISABLE_XPU_DEFAULT=1"]
    S1c["① qdf_mmff_predict.py<br/>--device cpu"]
    S2c["② compute_mmff_weights.py<br/>--source qdf --pred-csv ..."]
    S3c["③ compare_pretrain_ab.py<br/>A/B 마이크로벤치"]
    Ecpu --> S1c --> S2c --> S3c
  end

  subgraph XPU["MODE: xpu 디렉터리"]
    Expu["환경:<br/>TORCH_DEVICE=xpu<br/>(TORCH_DISABLE_XPU_DEFAULT 제거)"]
    S1x["① qdf_mmff_predict.py<br/>--device xpu"]
    S2x["② compute_mmff_weights.py"]
    S3x["③ compare_pretrain_ab.py"]
    Expu --> S1x --> S2x --> S3x
  end

  subgraph SHARED["공통 설정 예시"]
    MW["MMFF_WEIGHTS_PATH →<br/>각 모드별 mmff_weights.pt"]
    DL["DATALOADER_NUM_WORKERS=0<br/>PIN_MEMORY=0 기본"]
  end

  BOTH --> CPU
  BOTH --> XPU
  SHARED -.-> CPU
  SHARED -.-> XPU

  subgraph OUT["산출물"]
    M["manifest.json, steps.json"]
    P["compare_wall_by_step.png<br/>compare_resources.png"]
    L["cpu/logs/*.log, xpu/logs/*.log"]
    AB["각 모드/pretrain_ab/<br/>pretrain_ab_results.json, 막대·시계열 PNG"]
  end

  S3c --> OUT
  S3x --> OUT
  SKIP --> R0
  R0 --> R1
```

---

## 3. `compare_pretrain_ab.py` — 리소스 샘플링 (CPU % / XPU %)

사전학습 마이크로벤치 실행 중 **wall time**, **CPU·XPU 사용률**(가능 시 Windows PDH + QDF `bench/sampler.py`), **RSS** 등을 JSON/그림으로 남깁니다. A/B(또는 C) 구성은 `isolated_env()` 로 환경만 바꿔 **비파괴** 비교합니다.

```mermaid
sequenceDiagram
  participant Main as compare_pretrain_ab
  participant Core as pretrain_bench_core
  participant Sampler as QDF bench/sampler PDH
  participant Torch as PyTorch train step

  Main->>Core: run_pretrain_benchmark A
  Core->>Sampler: 구간 샘플링 (옵션)
  loop warmup + epochs
    Core->>Torch: forward / backward
    Sampler-->>Core: mean_cpu_pct, mean_xpu_pct, RSS…
  end
  Core-->>Main: summary_a
  Main->>Core: run_pretrain_benchmark B
  Core-->>Main: summary_b
  Note over Main: TORCH_DEVICE=cpu 이면<br/>XPU 카운터는 0에 가깝고,<br/>TORCH_DEVICE=xpu 이면<br/>XPU 활성 구간이 잡힘
```

---

## PNG 로보내기

- **권장:** [Mermaid Live](https://mermaid.live) 에서 위 블록을 붙여넣어 PNG/SVG 저장.
- **CLI:** `figures/xpu_cpu_orchestrator.mmd` 를 `@mermaid-js/mermaid-cli` 로 변환:

```text
npx --yes @mermaid-js/mermaid-cli -i figures/xpu_cpu_orchestrator.mmd -o figures/xpu_cpu_orchestrator.png -b white
```

오케스트레이터 요약도는 **`figures/xpu_cpu_orchestrator.png`** 로도 저장해 두었습니다(아래 `mmdc` 명령으로 재생성 가능).
