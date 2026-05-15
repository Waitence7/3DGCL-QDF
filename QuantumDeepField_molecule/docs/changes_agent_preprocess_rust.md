# 에이전트가 수정·추가한 부분만 (QDF 전처리 Rust / 벤치)

이 문서는 **이 대화에서 에이전트가 실제로 바꾼 코드·파일만** 나열합니다.  
`train/preprocess.py` 본문, ShardReader/Writer 등 **건드리지 않은 기존 코드는 적지 않습니다.**

---

## 1. `QuantumDeepField_molecule/qdf_io/src/lib.rs`

### 1.1 `preprocess_core` 안

| 변경 | 내용 |
|------|------|
| **`potential` 주석** | 레거시·테스트에서 쓰이고, 핫패스는 `potential_from_field_atoms`를 쓴다는 설명 3줄 추가(함수 본문 동일). |
| **`potential_from_field_atoms` (신규)** | `field`×`atomic_coords`×`atomic_numbers`로 가우시안 포텐셜 `(n_field,1)` 계산. `distance_matrix(field,atoms)` 없이 동일 수치 목표. `d==0`이면 `exp(-1e12)` 계열로 기존 `d=1e6` 가중과 맞춤. |
| **`process_one` (동작 변경)** | 기존: `dm_atoms = distance_matrix(...)` → `potential(&dm_atoms)` → `dm_orb`. **변경 후:** `potential_from_field_atoms(&field, ...)` → `dm_orb`만 `distance_matrix`. |
| **`process_one_legacy` (신규)** | 최적화 **이전**과 동일: `dm_atoms` + `potential(&dm_atoms)` + 궤도 `distance_matrix`. A/B·회귀용. |
| **`#[cfg(test)] mod tests` (신규/확장)** | (1) `potential_from_field_matches_distance_matrix_path` — 융합 vs `distance_matrix`+`potential`. (2) `process_one_matches_legacy_outputs` — `process_one` vs `process_one_legacy` dm/pot f32 허용차. |

### 1.2 PyO3 / 모듈 등록

| 변경 | 내용 |
|------|------|
| **`preprocess_batch_rust_legacy` (신규 `#[pyfunction]`)** | `preprocess_batch_rust`와 동일 인자·Rayon 구조, `par_iter`에서 `process_one_legacy`만 호출. |
| **`_native` 모듈 등록** | `m.add_function(wrap_pyfunction!(preprocess_batch_rust_legacy, m)?)?;` 한 줄 추가. |

**변경 없음:** `preprocess_molecule_rust`, `preprocess_batch_rust`(시그니처·입력 복사 패턴), `distance_matrix`/`create_field`/`to_f32` 본문, ShardReader/Writer 등 나머지 크레이트 대부분.

---

## 2. `QuantumDeepField_molecule/qdf_io/python/qdf_io/__init__.py`

| 변경 | 내용 |
|------|------|
| **`_load_native()`** | 패키지 `from . import _native` 성공 후 `ShardWriter` 있으면: **`preprocess_batch_rust_legacy` 속성이 없을 때** `sys.modules.pop` → `fresh_native_sidecar_path()` 및 `dev_native_dll_paths()` 순으로 `_try_exec_native` 재시도, legacy 있는 모듈이면 그걸 반환. 실패 시 다시 `from . import _native`. |
| **바인딩** | `preprocess_batch_rust_legacy = getattr(_mod, "preprocess_batch_rust_legacy", None)` (직접 속성 접근 대신). |
| **`__all__`** | 문자열 `"preprocess_batch_rust_legacy"` 항목 추가. |

**변경 없음:** `_try_exec_native`, `dev_native_dll_paths`, `fresh_native_sidecar_path` 시그니처; 다른 export 할당.

---

## 3. `QuantumDeepField_molecule/bench/profile_preprocess.py`

| 변경 | 내용 |
|------|------|
| **파일 상단 docstring** | `--backend rust-legacy` 예시 블록 추가. |
| **`--backend` choices** | `"rust-legacy"` 추가, help에 legacy 설명. |
| **`--rust-batch-size` help** | `rust / rust-legacy` 둘 다 언급. |
| **`rust_batch_fn` 도입** | `rust` / `rust-legacy`일 때 `qdf_io` import 후 `preprocess_batch_rust` vs `preprocess_batch_rust_legacy` 선택. |
| **`rust-legacy` + None** | `preprocess_batch_rust_legacy is None`이면 `sys.exit(...)`로 재빌드·DLL·Jupyter 잠금 안내. |
| **`flush_rust_chunk`** | `qdf_io.preprocess_batch_rust(...)` 고정 호출 → **`rust_batch_fn(...)`** 호출로 변경. |
| **루프 마지막 flush 조건** | `args.backend == "rust"` → **`args.backend in ("rust", "rust-legacy")`**. |
| **요약 출력** | `rust_batch_size` 출력 조건을 **`rust` 또는 `rust-legacy`** 로 확장. |

**변경 없음:** numpy 분기 전체, `parse_molecule` 측정, `timings` 키 이름(`rust_preprocess_batch` 등), `np.save` 경로.

---

## 4. `QuantumDeepField_molecule/bench/compare_preprocess_python_rust_before_after.ipynb`

**파일 전체가 신규 추가**에 가깝습니다. 셀 단위로 에이전트가 넣은 내용만 요약합니다.

| 셀/역할 | 내용 |
|---------|------|
| **Markdown(상단)** | 세 백엔드 표, subprocess cold start 설명, `RUN_LEGACY`, `qdf_io`/`pyd`/DLL 전제, **한국어 “한 작업 요약” 표** (Rust 최적·legacy·profile·로더·노트북). |
| **코드: 유틸** | `find_repo_root`, `parse_profile_preprocess`, `run_profile_preprocess`, `ensure_qdf_io_supports_rust_legacy`, `REPO` 출력. |
| **코드: 벤치** | `RUN_LEGACY`, `runs` 구성, `run_profile_preprocess` 루프, **`subprocess_elapsed_s` 컬럼**, per-run 및 총 driver 시간 `print`, `df` 표시. |
| **코드: wall 그래프** | 막대·색 팔레트·注釈·Rust-only `rust_preprocess_batch` 보조 그래프·legacy vs opt 비율 출력(백엔드로 행 선택). |
| **코드: 단계별 그래프** | `step__*` 컬럼으로 백엔드별 가로 막대 + union 단계 그룹 세로 막대. |

**다른 노트북·`train/preprocess.py`는 수정하지 않음.**

---

## 5. 문서 (신규 파일)

| 파일 | 내용 |
|------|------|
| `QuantumDeepField_molecule/docs/design_preprocess_rust_optimization_and_benchmark.md` | 설계 전반(배경·목표·다이어그램·운영 등). |
| **`QuantumDeepField_molecule/docs/changes_agent_preprocess_rust.md` (본 파일)** | **에이전트 diff만** 정리. |

---

## 6. 에이전트가 수정하지 않은 것 (명시)

- `QuantumDeepField_molecule/train/preprocess.py`
- `verify_preprocess_backend.py` (내용 변경 없음; 실행만 권장)
- `Cargo.toml` / `pyproject.toml` 의존성 버전
- `examples/sslgraph/bench/*` 일체
- 루트 `docs/3dgcl_work_summary_platforms.md` 등 기존 문서 본문

---

## 7. 한 줄 요약

**Rust:** 원자–필드 행렬 없이 포텐셜 계산 + 레거시 경로·배치 FFI·테스트.  
**Python:** 로더 DLL 폴백·`getattr(legacy)`·`profile_preprocess`에 `rust-legacy`·`rust_batch_fn`.  
**노트북:** 세 백엔드 비교·경과 시간·단계 그래프.  
**문서:** 설계 doc + 본 변경 전용 doc.
