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
- [examples/sslgraph/bench/compare_pretrain_ab.py](examples/sslgraph/bench/compare_pretrain_ab.py) — **비파괴** A/B: baseline vs optimized env 를 `isolated_env` 로만 적용·복구, QDF `bench/sampler.py` PDH 로 CPU/XPU/RSS 샘플링, JSON + 막대(bar wall·CPU·XPU·RSS, stage ms) + 시계열 CPU/XPU 그림을 `bench/figs/pretrain_ab_<ts>/` 에 저장.
- [examples/sslgraph/bench/run_pretrain_compare.ipynb](examples/sslgraph/bench/run_pretrain_compare.ipynb) — 위 스크립트를 실행하고 생성된 PNG·JSON 요약을 노트북에서 표시.

## 그 외

- [dig/sslgraph/utils/importtest.ipynb](dig/sslgraph/utils/importtest.ipynb) — `dig` 패키지의 `SphereNet`/`DimeNetPP` import 경로가 살아 있는지 확인하는 한 줄짜리 import 테스트.
- [examples/Untitled.ipynb](examples/Untitled.ipynb) — 빈 스크래치 파일(코드 셀 없음). 실행할 내용 없음.

---

### 참고

- 위 노트북 중 fig 를 저장하는 것들(`bench/run_*.ipynb`, `tst.ipynb`)은 모두 실행 시각 `FIG_TS = strftime("%m%d_%H%M")` 를 파일명 끝에 붙입니다. 재실행해도 기존 그림을 덮어쓰지 않습니다.
- 인터프리터는 일관되게 `c:\DGCL\3DGCL\.venv\Scripts\python.exe` 사용 (Python 3.10 + Torch 2.9.1+xpu 기준).
- `.ipynb_checkpoints/` 아래 자동 백업본은 인덱스에서 제외.
