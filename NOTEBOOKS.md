# 노트북 인덱스

레포 안의 실행 가능한 `.ipynb` 파일과 각 노트북의 역할 한 줄 요약. 모든 경로는 레포 루트 (`c:\DGCL\3DGCL`) 기준 상대경로.

## QDF (`QuantumDeepField_molecule/`) — Rust 통합·전처리·예측 파이프라인

- [QuantumDeepField_molecule/bench/run_pipeline.ipynb](QuantumDeepField_molecule/bench/run_pipeline.ipynb) — QDF Rust 풀-파이프라인: 환경 확인 → `qdf_io` 빌드 → 전처리(`numpy` vs `rust`) → npy↔shard 변환 → 학습 프로파일(`--loader`×`--pad-impl`) → 막대/시계열/CPU·XPU 비교 그림 저장.
- [QuantumDeepField_molecule/bench/run_predict.ipynb](QuantumDeepField_molecule/bench/run_predict.ipynb) — QDF 추론 전용 비교: `profile_predict.py` 호출로 `numpy` 백엔드 vs Rust 경로 wall/RAM/CPU%·XPU% 측정 + `predict_*.png` 저장.
- [QuantumDeepField_molecule/tst.ipynb](QuantumDeepField_molecule/tst.ipynb) — 학습 로그(`output/resulthomo.txt`)에서 epoch별 MAE_val/MAE_test 파싱·플롯하는 스크래치 노트북.

## DGCL (`examples/sslgraph/`) — 자기지도 사전학습·미세조정 + Rust 벤치

- [examples/sslgraph/pretrain.ipynb](examples/sslgraph/pretrain.ipynb) — SchNet/SphereNet 등 인코더를 ESOL/FreeSolv/Lipo 같은 데이터셋에서 GraphCL/InfoGraph 로 사전학습. XPU/CUDA 자동 감지.
- [examples/sslgraph/finetune.ipynb](examples/sslgraph/finetune.ipynb) — 사전학습 가중치(`args.model_path`) 로드 후 분자 회귀 grid-search; 평균 test RMSE 시각화 + 결과 CSV 저장.
- [examples/sslgraph/downstream.ipynb](examples/sslgraph/downstream.ipynb) — `args.finetune=False` 로 동일 grid-search 를 "random init" baseline 으로 돌려 finetune 결과와 비교.
- [examples/sslgraph/bench/run_dgcl_rust.ipynb](examples/sslgraph/bench/run_dgcl_rust.ipynb) — DGCL Rust(`dig_io`) 통합 검증·벤치: 환경 점검 → (선택) 빌드 → `verify_views_backend` → view fn(Uniform/RW/EdgePerturb) Python vs Rust 마이크로벤치 → `key_split` 정합성(size/set/disjoint) → `MoleculeNet` vs `MoleculeNetShard` 로드·이터레이션 → `verify_dig_io_fallback` → 결과 표·막대그래프(`bench/figs/bars_views_{ts}.png`).
- 동반 스크립트 [examples/sslgraph/bench/profile_pretrain.py](examples/sslgraph/bench/profile_pretrain.py) — pretrain 한 epoch 을 dataloader/views_fn/forward/backward/optim 단계로 쪼개 wall time 측정. 환경 변수 `MMFFRANDOM_FAST=1`, `PRETRAIN_AMP=bf16`, `DATALOADER_NUM_WORKERS=N`, `PIN_MEMORY=1` 조합으로 효과 검증 가능.
- [examples/sslgraph/bench/compare_pretrain_ab.py](examples/sslgraph/bench/compare_pretrain_ab.py) — **비파괴** A/B/C 비교: baseline vs optimized vs (선택) weighted env 를 `isolated_env` 로만 적용·복구, QDF `bench/sampler.py` PDH 로 CPU/XPU/RSS 샘플링. JSON + 막대(wall·CPU·XPU·RSS, stage ms — **값 라벨 박힘**) + 시계열 CPU/XPU 그림을 `bench/figs/pretrain_ab_<ts>/` 에 저장. C run 활성화: `--c-label/--c-set/--c-clear` (예: `--c-set MMFFRANDOM_WEIGHTED=1 --c-set MMFF_WEIGHTS_PATH=dataset/esol_mmff_weights_qdf.pt`).
- [examples/sslgraph/bench/run_pretrain_compare.ipynb](examples/sslgraph/bench/run_pretrain_compare.ipynb) — 위 스크립트를 실행하고 결과를 표시. `COMPARE_MODE = "abc"` + `WEIGHT_SOURCE = "boltzmann"|"qdf"` 토글로 가중치 `.pt` 가 없으면 자동 빌드 (QDF 모드는 `qdf_mmff_predict.py` → `compute_mmff_weights.py` 체인).
- [examples/sslgraph/bench/compare_pretrain_quality.ipynb](examples/sslgraph/bench/compare_pretrain_quality.ipynb) — **품질** 비교 (속도 X). A/B/C 동일 하이퍼파라미터로 GraphCL 풀-프리트레인 + ESOL finetune (scaffold 3-fold) → test RMSE 막대 + loss 곡선. 추가로 **슬롯 분포 검증**: 각 view 함수를 1 배치에 N 번 호출해 MMFF1..4 슬롯 선택 빈도 (`slot_freq`, `KL(uniform)`, per-graph entropy) 를 그려 \"진짜 랜덤이 weighted 로 바뀌었는지\" 확인. 결과는 `bench/figs/pretrain_quality_<ts>/` 와 `models/quality_<ts>/{A,B,C}/`.
- 동반 모듈 [examples/sslgraph/bench/pretrain_quality_core.py](examples/sslgraph/bench/pretrain_quality_core.py) — quality 노트북이 사용하는 헬퍼: `run_pretrain_side(side, ...)`, `run_finetune_side(...)`, `measure_slot_distribution(side, ...)`. 측면별 `MMFFRANDOM_FAST/WEIGHTED` 환경 변수는 `_clean_aug_env()` 가 자동 마스킹 (호스트 env 비파괴).
- 동반 유틸 [examples/sslgraph/bench/dump_mmff_weights.py](examples/sslgraph/bench/dump_mmff_weights.py) — `compute_mmff_weights.py` 가 만든 `.pt` (분자→slot 가중치) 를 사람이 읽을 수 있는 CSV 로 변환 (`smiles, w1..w4, top_slot, entropy_bits, top_minus_uniform, source`). `--sort-by top_minus_uniform` 로 가장 한쪽 슬롯에 쏠린 분자 먼저 보기.
- [examples/sslgraph/bench/compute_mmff_weights.py](examples/sslgraph/bench/compute_mmff_weights.py) — `MMFFweighted` 뷰가 사용할 per-(smiles, slot) 가중치 사전 계산. `--source boltzmann` (기본, MoleculeNet `maxK_energy` 로 즉시) 또는 `--source qdf --pred-csv` (CSV 컬럼으로 schema 자동 감지: `homo,lumo` ⇒ HOMO-LUMO gap softmax / `energy` ⇒ atomization Boltzmann; `--score-expr` 로 표현식 오버라이드, `--normalize zscore` 기본). 결과는 `dataset/<name>_mmff_weights_<source>.pt` (메타: `qdf_property`, `score_expr`, `normalize`).
- [examples/sslgraph/bench/qdf_mmff_predict.py](examples/sslgraph/bench/qdf_mmff_predict.py) — DGCL MoleculeNet 의 분자 × MMFF 슬롯(1–4) 좌표를 **QDF 체크포인트**로 추론해 `dataset/<name>_qdf_mmff_preds_<property>.csv` 생성. `--qdf-property atomization` (기본, **conformer-sensitive**, QM9under14atoms ckpt) 또는 `--qdf-property homolumo` (legacy, QM9under7atoms ckpt). 아키텍처 (dim/op/hidden_HK) 와 ckpt 경로는 property preset 으로 자동 결정 — `--checkpoint` 로 다른 ckpt 시도 가능. orbital_dict 미등록 원소(P/S/Cl/Br/I 등) 자동 스킵 → downstream Boltzmann fallback. ESOL XPU 기준 ~1.5분.

## 그 외

- [dig/sslgraph/utils/importtest.ipynb](dig/sslgraph/utils/importtest.ipynb) — `dig` 패키지의 `SphereNet`/`DimeNetPP` import 경로가 살아 있는지 확인하는 한 줄짜리 import 테스트.
- [examples/Untitled.ipynb](examples/Untitled.ipynb) — 빈 스크래치 파일(코드 셀 없음). 실행할 내용 없음.

---

### 참고

- 위 노트북 중 fig 를 저장하는 것들(`bench/run_*.ipynb`, `tst.ipynb`)은 모두 실행 시각 `FIG_TS = strftime("%m%d_%H%M")` 를 파일명 끝에 붙입니다. 재실행해도 기존 그림을 덮어쓰지 않습니다.
- 인터프리터는 일관되게 `c:\DGCL\3DGCL\.venv\Scripts\python.exe` 사용 (Python 3.10 + Torch 2.9.1+xpu 기준).
- `.ipynb_checkpoints/` 아래 자동 백업본은 인덱스에서 제외.
