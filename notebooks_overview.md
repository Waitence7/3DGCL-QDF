# 3DGCL 노트북 전체 정리

> 이 문서는 `/home/ubuntu/3DGCL` 레포지터리 내 모든 Jupyter 노트북의 **의도 · 결과 · 연관 관계**를 정리한 색인입니다.  
> 최종 갱신: 2026-05-15

---

## 전체 구조 한눈에 보기

```
3DGCL/
├── QuantumDeepField_molecule/
│   ├── tst.ipynb                              ← 로그 파싱 & MAE 플롯 (스크래치)
│   └── bench/
│       ├── run_pipeline.ipynb                 ← QDF 전처리 + 학습 파이프라인 비교
│       ├── run_pipeline copy.ipynb            ← run_pipeline의 사용자 실험 사본
│       ├── run_predict.ipynb                  ← QDF 추론 경로 비교
│       ├── run_predict copy.ipynb             ← run_predict의 사용자 사본
│       ├── run_predict copy.executed.ipynb    ← 실행 결과 저장본
│       ├── ensemble_comparison.ipynb          ← E-only vs E+V 앙상블 비교
│       └── ensemble_comparison.executed.ipynb ← 앙상블 비교 실행 결과 저장본
│
├── examples/sslgraph/
│   ├── pretrain.ipynb                         ← GraphCL 사전학습 (ESOL)
│   ├── finetune.ipynb                         ← 사전학습 체크포인트로 파인튜닝
│   ├── downstream.ipynb                       ← 다운스트림 baseline/pretrained 비교
│   └── bench/
│       ├── run_pretrain_compare.ipynb         ← Pretrain A/B/C 속도 비교
│       ├── compare_pretrain_quality.ipynb     ← Pretrain A/B/C 품질(RMSE) 비교
│       ├── compare_pretrain_quality_homolumo.ipynb  ← homolumo 가중치 품질 비교
│       ├── _smoke_out.ipynb                   ← smoke 테스트용 미니 노트북
│       └── run_dgcl_rust.ipynb                ← DGCL dig_io Rust 벤치
│
├── dig/sslgraph/utils/
│   └── importtest.ipynb                       ← SphereNet/DimeNetPP import 디버깅
│
├── examples/ (기타)
│   └── Untitled.ipynb                         ← 빈 노트북
│
└── test/                                      ← 위 노트북들의 테스트 트리 사본
    ├── QuantumDeepField_molecule/bench/...
    ├── examples/sslgraph/bench/...
    │   ├── rust_speed_efficiency_report.ipynb
    │   └── rust_parallel_tuning_report.ipynb
    └── dig/sslgraph/utils/...
```

---

## 1. QDF (QuantumDeepField) 파이프라인

### 1-1. `QuantumDeepField_molecule/bench/run_pipeline.ipynb`

**의도**  
QDF 전처리(NumPy vs Rust)와 학습(loader × pad_impl 4가지 조합)을 하나의 노트북에서 순서대로 실행하고, `bench/sampler.py`로 wall time · RSS · CPU% · XPU%를 수집해 그래프로 보여 주는 **전체 파이프라인 통합 벤치마크**.

**실행 순서**
1. 환경 점검 (`torch`, `qdf_io`, XPU 인식)
2. (선택) Rust 확장 빌드
3. 전처리 실행 — `preprocess.py --backend {numpy,rust}`
4. 전처리 결과 검증 — `verify_preprocess_backend.py`
5. Shard 변환 — `convert_to_shard.py`
6. Shard 검증 — `verify_shard.py`
7. 학습 프로파일 — `profile_train.py` (4가지 조합)
8. LCAO 패치 검증 — `verify_lcao_patch.py`
9. 전처리 마이크로벤치 — `profile_preprocess.py`
10. `sampler.py` 통합 측정 + 시각화 저장 (`bench/figs/`)

**주요 결과**

| 구성 | 전체 wall (60 batch × 2 epoch) |
|------|-------------------------------|
| `npy` + `python` | ~50.3 s |
| `shard` + `python` | ~49.1 s |
| `shard` + `rust-pad-only` | ~30.8 s |
| `shard` + `rust` | ~38.7 s |

- 전처리: NumPy **~8.7 s** vs Rust **~0.37 s** (1000분자, `--no-save`) — **≈22x 가속**
- Shard 정확성: 200분자 `rtol=1e-5, atol=1e-5` 이내 완전 일치
- LCAO 패치 정확성: python vs rust-pad-only vs rust — **loss(E/V) 완전 동일**

---

### 1-2. `QuantumDeepField_molecule/bench/run_pipeline copy.ipynb`

**의도**  
`run_pipeline.ipynb`의 사용자 실험 사본. 기본 흐름은 동일하며, `§0 공통 설정`에 배포 데이터셋 목록(`QM9under7atoms_*`, `QM9full_homolumo_eV`)과 수동 준비가 필요한 데이터셋(`QM9under14atoms_atomizationenergy_eV`) 안내가 추가되어 있다. `§10.3 CPU 스레드 스윕` 섹션을 포함한 확장판.

**주요 결과**  
실행 결과는 `run_pipeline.ipynb`과 동일 수준이며, 그림은 `bench/figs/{MMDD_HHMM}_{N_CPUS}cpu/` 폴더에 저장.

---

### 1-3. `QuantumDeepField_molecule/bench/run_predict.ipynb`

**의도**  
학습 경로가 아닌 **추론(inference) 경로만** 비교하는 노트북. `bench/profile_predict.py`를 NumPy vs Rust 4가지 조합으로 실행하고, `sampler.py`로 리소스를 측정한다. 전처리 / Shard는 `run_pipeline.ipynb` 결과물을 재사용.

**기본 시나리오**: `QM9full_homolumo_eV`, `mean`, `test` split

**주요 결과** (`QM9under14atoms_atomizationenergy_eV`, `sum`, batch 8, 60 batches 기준)

| 구성 | Wall | Speedup |
|------|------|---------|
| `npy` + `python` | 9.804 s | ×1.00 |
| `shard` + `python` | 4.654 s | ×2.11 |
| `shard` + `rust-pad-only` | 5.086 s | ×1.93 |
| `shard` + `rust` | 4.627 s | ×2.12 |

그림 저장: `bench/figs/predict_{bars,timeseries,cpu_xpu}.png`

---

### 1-4. `QuantumDeepField_molecule/bench/run_predict copy.ipynb` / `run_predict copy.executed.ipynb`

**의도**  
`run_predict.ipynb`의 사용자 사본. `.executed` 버전은 특정 실행의 결과를 그대로 저장한 **실행 아티팩트**.

**주요 결과**  
Phase A(데이터로더 단독) 기준: `npy` **4.449 s** → `shard+python` **0.290 s** → `shard+rust` **~0.09 s** 수준.

---

### 1-5. `QuantumDeepField_molecule/bench/ensemble_comparison.ipynb`

**의도**  
QDF의 **이중 태스크(E + V / HK map) 앙상블**이 E-only 단일 태스크 대비 얼마나 성능이 개선되는지 비교.  
비교 대상: ① Random(mean) baseline, ② Untrained QDF, ③ E-only, ④ E+V (full ensemble).

**주요 결과** (소규모 QM9 atomization, 10 epoch)

| 모델 | Test MAE |
|------|----------|
| Random (mean) baseline | 17.16 eV |
| Untrained QDF | — |
| E-only | **5.17 eV** |
| E+V ensemble | **5.16 eV** (+0.2% 개선) |

노트 : 10 epoch은 수렴 전 단계이므로, 장기 학습 시 개선폭이 더 커질 것으로 예상.

---

### 1-6. `QuantumDeepField_molecule/bench/ensemble_comparison.executed.ipynb`

**의도**  
`ensemble_comparison.ipynb`의 **다른 설정(데이터셋 · 에폭 수)으로 실행한 결과 저장본**.

**주요 결과**

| 모델 | Test MAE |
|------|----------|
| E-only | 2.0813 eV |
| E+V ensemble | 2.0778 eV (+0.2% 개선) |

---

### 1-7. `QuantumDeepField_molecule/tst.ipynb`

**의도**  
학습 로그 파일(`output/resulthomo.txt`)에서 `MAE_val` / `MAE_test`를 정규식으로 파싱해 matplotlib으로 시각화하고 PNG로 저장하는 **임시 스크래치 노트북**.

**특이사항**  
`C:/QuantumDeepField_molecule/...` 경로를 하드코딩 — Windows 로컬 환경 전용. 재사용 시 경로 수정 필요.

**결과**  
`MAE_val_plot.png`, `MAE_test_plot.png` 생성 (50 epoch 파싱 예시).

---

## 2. DGCL (GraphCL + sslgraph)

### 2-1. `examples/sslgraph/pretrain.ipynb`

**의도**  
ESOL 데이터셋에 SchNet 인코더 + SphereNet 프로젝션으로 **GraphCL 사전학습**을 수행하는 메인 노트북.  
`pick_torch_device`로 XPU/CUDA/CPU를 자동 선택하며, 환경변수(`MMFFRANDOM_FAST`, `PIN_MEMORY` 등)로 최적화 경로를 토글할 수 있다.

**주요 결과**  
- Device: `xpu:0` (Intel Arc 140V)
- 학습 수렴 예시: epoch 82 기준 best loss **9.112**, 체크포인트 `enc_best_epoch-82_loss-9.112.pkl` 저장

---

### 2-2. `examples/sslgraph/finetune.ipynb`

**의도**  
사전학습된 체크포인트를 불러와 ESOL 회귀 태스크로 **파인튜닝** 및 `Finetune.grid_search` 실행.

**주요 결과**  
- Test RMSE: **~0.8959** (grid 평균 ~0.8876)
- `predictions_best_grid.csv` 저장 (`y_true`, `y_pred`, `abs_error`)

---

### 2-3. `examples/sslgraph/downstream.ipynb`

**의도**  
파인튜닝 결과를 검증하는 노트북. `args.finetune = False`로 **랜덤 초기화 baseline**도 함께 비교 가능.

**주요 결과**  
K-fold early stopping 결과 저장, `predictions_best_grid.csv` 출력. 일부 스테로이드 분자에서 오차가 크게 나타남.

---

### 2-4. `examples/sslgraph/bench/run_pretrain_compare.ipynb`

**의도**  
사전학습 **속도 · 리소스** 비교를 위한 A/B/C 벤치마크 노트북.  
`bench/compare_pretrain_ab.py` + `sampler.py` 조합으로 측정하며, 호스트 환경을 오염시키지 않는 `isolated_env()` 패턴 사용.

| 측면 | 구성 |
|------|------|
| A | 기존 PyG 경로 (`RandomView`) |
| B | `MMFFRANDOM_FAST=1` + AMP bf16 |
| C | QDF/Boltzmann 가중 슬롯 + `.pt` weights |

**주요 결과** (ESOL, batch 400, 1 epoch × 2 iters)

| 구성 | Wall | mean CPU% | mean XPU% |
|------|------|-----------|-----------|
| A baseline | 5.53 s | 86.4% | 37.9% |
| B optimized | 1.74 s | 96.8% | 53.4% |
| C weighted (QDF) | **1.68 s** | 96.3% | 48.9% |

**결론**: views_fn 병목(A의 60%)을 `FastRandomMMFFView`(torch 벡터화)로 해결 → **3.2× 가속**.  
그림 저장: `figs/pretrain_ab_{ts}/`

---

### 2-5. `examples/sslgraph/bench/compare_pretrain_quality.ipynb`

**의도**  
속도 비교가 아닌 **다운스트림 RMSE + 슬롯 분포 검증**을 담당하는 품질 비교 노트북.  
`pretrain_quality_core.py`를 통해 A/B/C 각 측면을 독립적으로 사전학습 후 파인튜닝 RMSE 측정.

**슬롯 분포 검증 결과** (ESOL, batch 256, 10 calls)

| 측면 | view | slots_used | KL(uniform) | per-graph H̄ |
|------|------|-----------|-------------|--------------|
| A baseline | `RandomView` | 2 | 1.000 b | 0.980 b |
| B fast | `FastRandomMMFFView` | 2 | 1.000 b | 0.981 b |
| C weighted (수정 후) | `WeightedMMFFView` | 4 | 0.000 b | **1.643 b** |

**Finetune RMSE** (대형 실행 기준)

| 측면 | RMSE |
|------|------|
| A | 0.8424 ± 0.1526 |
| B | 1.6692 ± 0.4094 |
| C | 1.6692 ± 0.4094 |

노트: B/C가 A보다 높은 RMSE가 나온 실행 — 설정(`QUALITY_SMOKE: True`, `gin` 인코더)이 다른 조건임을 주의. 정식 비교는 전체 에폭 + schnet 인코더로 재실행 권장.

그림 저장: `figs/pretrain_quality_{ts}/`

---

### 2-6. `examples/sslgraph/bench/compare_pretrain_quality_homolumo.ipynb`

**의도**  
`compare_pretrain_quality.ipynb`의 **homolumo 가중치 변형**. `QDF_PROPERTY='homolumo'`로 HOMO-LUMO gap을 점수로 슬롯 가중치를 만들고 품질을 비교.  
`QUALITY_BUDGET` 노브(`smoke`/`two_hour`/`max_accuracy`)로 실험 규모 조정 가능.

**특이사항**  
HOMO-LUMO 기반 가중치는 conformer 간 spread가 너무 작아(~1 meV, 분자 간 절대값 대비 5×10⁻⁴) 정보량이 부족함 → atomization energy 기반(`compare_pretrain_quality.ipynb`)이 권장.

---

### 2-7. `examples/sslgraph/bench/_smoke_out.ipynb`

**의도**  
`compare_pretrain_quality`의 **최소 smoke 실행용** 발췌 노트북. intro markdown + 설정 셀만으로 구성되어 빠른 파이프라인 점검에 사용.

---

### 2-8. `examples/sslgraph/bench/run_dgcl_rust.ipynb`

**의도**  
DGCL 측 Rust 확장(`dig_io`)의 **검증 + 벤치마크** 통합 노트북.

**실행 순서**
1. 환경 점검 (`dig_io` 임포트, `is_available()`)
2. (선택) `dig_io` 빌드
3. View fn 정확성 검증 — `verify_views_backend.py`
4. View fn 마이크로벤치 (Python vs Rust)
5. `key_split` / `scaffold_split` 정합성 검증
6. `MoleculeNet` → `MoleculeNetShard` 변환 및 로드 벤치 (데이터셋 있을 때만)
7. fallback 검증 — `verify_dig_io_fallback`
8. 결과 표 + 막대그래프

**주요 결과** (CPU 1코어, 합성 배치 n_graphs=128)

| 연산 | Python ms/call | Rust ms/call | 배율 |
|------|---------------|-------------|------|
| `UniformSample` | 16.72 | 9.51 | **1.76×** |
| `RWSample(0.5)` | 57.28 | 25.48 | **2.25×** |
| `EdgePerturbation` | 59.91 | 9.01 | **6.65×** |

그림 저장: `examples/sslgraph/bench/figs/bars_views_{ts}.png`

---

## 3. 유틸리티 / 디버깅

### 3-1. `dig/sslgraph/utils/importtest.ipynb`

**의도**  
`dig/sslgraph/utils/`에 벤더링된 `SphereNet` / `DimeNetPP` 모듈의 **import 경로 디버깅** 노트북.

**결과**  
- `SphereNet` import 시 `ModuleNotFoundError: No module named 'features'` 발생
- `DimeNetPP` import 시 `ValueError: attempted relative import beyond top-level package` 발생
- → `sys.path` 조정이나 패키지 구조 수정 필요

---

### 3-2. `examples/Untitled.ipynb`

**의도 / 결과**: 빈 노트북. 내용 없음.

---

## 4. test/ 트리 사본

아래 노트북들은 `test/` 디렉터리 아래에 위치하며, 각자 대응하는 원본과 동일한 내용을 가집니다. CI 또는 격리된 환경 검증용으로 존재합니다.

| test/ 경로 | 대응 원본 |
|-----------|----------|
| `test/QuantumDeepField_molecule/bench/run_pipeline.ipynb` | `QuantumDeepField_molecule/bench/run_pipeline.ipynb` |
| `test/QuantumDeepField_molecule/bench/run_predict.ipynb` | `QuantumDeepField_molecule/bench/run_predict.ipynb` |
| `test/examples/sslgraph/pretrain.ipynb` | `examples/sslgraph/pretrain.ipynb` |
| `test/examples/sslgraph/bench/compare_pretrain_quality.ipynb` | `examples/sslgraph/bench/compare_pretrain_quality.ipynb` |
| `test/examples/sslgraph/bench/run_pretrain_compare.ipynb` | `examples/sslgraph/bench/run_pretrain_compare.ipynb` |
| `test/examples/sslgraph/bench/run_dgcl_rust.ipynb` | `examples/sslgraph/bench/run_dgcl_rust.ipynb` |
| `test/dig/sslgraph/utils/importtest.ipynb` | `dig/sslgraph/utils/importtest.ipynb` |

### 4-1. `test/examples/sslgraph/bench/rust_speed_efficiency_report.ipynb`

**의도**  
`연구노트.md`의 Rust 속도/효율 관련 결과를 **재현 가능한 보고서** 형태로 정리한 노트북.  
`profile_preprocess.py`, `profile_views.py` 결과를 묶고, LCAO Rust ROI 및 pretrain 병목 분석을 포함. `rust_parallel_tuning_report.ipynb`로 연결됨.

---

### 4-2. `test/examples/sslgraph/bench/rust_parallel_tuning_report.ipynb`

**의도**  
Rayon 스레드 수(`RAYON_NUM_THREADS`), `--rust-batch-size`, DataLoader `num_workers` 조합의 **병렬화 튜닝** 보고서 템플릿.  
대부분의 셀이 `execution_count: null` — **로컬 실행 후 결과 채워 넣는 용도**.

---

## 5. 노트북 간 실행 순서 및 의존 관계

```
[전처리]
  run_pipeline.ipynb
    └─ preprocess.py (numpy/rust)
    └─ convert_to_shard.py
    └─ verify_shard.py
    └─ profile_train.py
    └─ profile_preprocess.py
    └─ sampler.py

[추론]
  run_predict.ipynb          (run_pipeline의 shard 결과 재사용)
    └─ profile_predict.py
    └─ sampler.py

[사전학습 속도]
  run_pretrain_compare.ipynb
    └─ compare_pretrain_ab.py
    └─ pretrain_bench_core.py
    └─ sampler.py (QDF 스타일)

[사전학습 품질]
  compare_pretrain_quality.ipynb
    └─ pretrain_quality_core.py
    └─ qdf_mmff_predict.py        (가중치 미존재 시 자동 호출)
    └─ compute_mmff_weights.py    (가중치 미존재 시 자동 호출)
    └─ (run_pretrain_compare.ipynb의 A/B/C 정의 공유)

[DGCL Rust]
  run_dgcl_rust.ipynb
    └─ dig_io (Rust 빌드)
    └─ verify_views_backend.py
    └─ profile_views.py
    └─ profile_dataset_build.py
```

---

## 6. 핵심 수치 요약

| 항목 | NumPy/Python | Rust/Shard | 배율 |
|------|-------------|-----------|------|
| 전처리 (1000분자, compute-only) | 8.73 s | 0.37 s | **≈22×** |
| 데이터로더 Phase A (batch당) | 30.4 ms | 8.6 ms | **3.5×** |
| 학습 전체 wall (60 batch × 2 epoch) | ~50 s | ~31–39 s | **1.3–1.6×** |
| 추론 전체 wall (60 batch) | 9.8 s | 4.6 s | **2.1×** |
| EdgePerturbation view fn | 59.9 ms | 9.0 ms | **6.6×** |
| pretrain 1 epoch (views 병목 해결 후) | 6106 ms (A) | 2070 ms (B+AMP) | **3.0×** |
| LCAO pad+list_to_batch (Rust) | python 기준 | 동등~느림 | **비권장** |

---

## 7. 제약 및 주의사항

- **`tst.ipynb`**: `C:/QuantumDeepField_molecule/` 하드코딩 — Windows 전용, 재사용 시 경로 수정 필요.
- **LCAO Rust (`--pad-impl rust`)**: Intel Arc 140V + batch 8 환경에서는 Python 대비 느리거나 같음. 기본값은 Python 유지; 다른 GPU/배치 크기에서 재평가 권장.
- **Cold cache 벤치**: 현재 측정은 대부분 OS 캐시 warm 상태. Cold cache 격차는 별도 측정 필요 (RAMMap 등 사용).
- **`compare_pretrain_quality`의 B/C RMSE**: `QUALITY_SMOKE=True` + `gin` 인코더 조건 결과. 정식 비교는 `schnet` + 충분한 에폭 필요.
- **ESOL 일부 SMILES**: trailing whitespace가 있어 SMILES 매칭 시 `.strip()` 처리 필요 (이미 반영됨).
- **`.venv` 내 RDKit 노트북**: upstream 테스트 파일로, 이 목록에서 제외.
