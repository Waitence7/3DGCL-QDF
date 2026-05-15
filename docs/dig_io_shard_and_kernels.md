# DGCL Rust (`dig_io`) — shard·뷰 커널·스캐폴드

이 문서는 레포의 **`dig_io`** 모듈이 무엇을 하고, **shard**가 QDF의 shard와 어떻게 다른지 정리합니다.

**도표만 (SVG/PNG):** [`figures/dig_io_shard_figure.svg`](figures/dig_io_shard_figure.svg) · [`figures/dig_io_shard_figure.png`](figures/dig_io_shard_figure.png) · 인덱스 [`figures/dig_io_shard_figure.md`](figures/dig_io_shard_figure.md)

---

## 1. `dig_io`가 하는 일

`dig_io`는 `dig_io/src/lib.rs` 머리 주석대로 **`qdf_io`와 같은 스택**(PyO3 + numpy + rayon + memmap2)이지만, **DGCL(그래프·대조학습) 파이프라인**에 맞는 네이티브 도우미만 노출합니다.

크게 **세 덩어리**입니다.

| 블록 | 역할 | Python에서 쓰는 곳(예) |
|------|------|-------------------------|
| **대조 뷰(view) 커널** | `edge_index` 등을 받아 **결정적(seed) 서브그래프 샘플링** 등을 CPU에서 빠르게 처리, numpy로 반환 | GraphCL 쪽 view 구현이 Rust 경로로 호출 가능 |
| **스캐폴드 분할** | `scaffold_bucket_split`, `scaffold_bucket_sort` — **버킷 경계·정렬**을 Rust로 | `dig/threedgraph/dataset/dataset.py`의 `key_split(..., impl='rust')` |
| **Molecule shard** | `MoleculeShardWriter` / `MoleculeShardReader` — **PyG `Data`와 동등한 정보를 한 파일에 패킹** | `MoleculeNetShard.py`, `examples/sslgraph/convert_dataset_to_shard.py` |

`dig_io/python/dig_io/__init__.py`에 **`_REQUIRED_SYMBOLS`**로 위 심볼들이 전부 있어야 “로드 성공”으로 칩니다. 하나라도 없으면 `is_available()`이 거짓이 됩니다.

---

## 2. “Shard”를 두 종류로 나눠 이해하기 (QDF vs DGCL)

같은 단어 **shard**지만 **완전히 다른 바이너리 포맷**입니다.

| | **QDF (`qdf_io`)** | **DGCL (`dig_io`)** |
|--|-------------------|---------------------|
| **매직** | `QDFSHRD\0` | `DIGSHRD\0` |
| **담는 것** | QDF 전처리 한 분자: 필드 격자·거리행렬·포텐셜 등 **3D 필드 학습용 레코드** | MoleculeNet이 만든 **PyG `Data` 한 개**에 대응: `z, pos, edge_index, …`, (선택) **MMFF 4슬롯 좌표·에너지** |
| **옆에 두는 파일** | 보통 `train_*_shard.bin` 등 (전처리 출력 트리) | **`{root}/{name}/processed/data.pt` 옆의 `data.shard`** |
| **읽기 방식** | `qdf_io.ShardReader` + Python이 `np.load` 등으로 이어 읽기 | **`mmap`**으로 인덱스 테이블 따라가며 레코드 단위로 디코드 |
| **왜 쓰나** | `.npy` 수십만 개·느린 IO 완화 | **`data.pt` 전체를 메모리에 올리는 부담** 완화, 로더 I/O 개선 |

즉, **“shard = mmap 단일 파일”이라는 아이디어는 비슷하지만, 파일 포맷·필드·소비 코드가 전부 다릅니다.** QDF 파이프라인 문서와 섞이면 헷갈리기 쉽습니다.

---

## 3. DGCL shard (`DIGSHRD`) 레이아웃

`dig_io/src/lib.rs` 상단 주석에 **바이트 단위 스펙**이 적혀 있습니다. 요지만 말하면:

1. **64B 헤더**  
   - 매직 `DIGSHRD\0`, version, `n_records`, 인덱스/데이터 섹션 오프셋, `file_size` 등  
2. **인덱스 테이블**  
   - 분자(레코드) `i`의 시작 위치를 가리키는 **절대 오프셋 `u64` 배열**  
3. **데이터 섹션 — 레코드마다**  
   - 짧은 **레코드 헤더**(원자 수, 엣지 수, `edge_attr` 폭, `x` 폭, `y` 차원, idx/smiles 길이, **MMFF 존재 플래그** 등)  
   - UTF-8 `idx`, `smiles` (8바이트 정렬 패딩)  
   - `z` (i64), `pos` (f32), `edge_index` (i64), `edge_attr`, `x`, `y`  
   - **MMFF가 있으면** `max1..4pos_mmff` 각 `(n_atoms*3)` float + 에너지 스칼라들(f64) 등 (소스 주석에 순서 명시)

`dig/threedgraph/dataset/MoleculeNetShard.py`의 `_data_to_record` / `_record_to_data`가 **“PyG `Data` ↔ 이 레코드 dict”** 변환을 맞춰 둔 쪽입니다. 즉 **shard는 “이미 process된 MoleculeNet 텐서들의 직렬화”**에 가깝습니다.

---

## 4. Python 쪽 사용 흐름

1. **먼저** 보통 `MoleculeNet`이 `processed/data.pt`를 만듭니다.  
2. **`convert_inmemory_to_shard`** (`MoleculeNetShard.py`) 또는 CLI **`examples/sslgraph/convert_dataset_to_shard.py`**로  
   - 메모리/디스크上的 `Dataset[i]`를 순회하며  
   - `dig_io.MoleculeShardWriter.append_record(...)`로 **`data.shard`**를 생성합니다.  
3. 학습/파인튜닝에서 **`loader='shard'`**를 고르면 (`dig/sslgraph/evaluation/finetune.py` 등)  
   - `default_shard_path(root, name)` → `{root}/{name}/processed/data.shard`  
   - **`MoleculeNetShard`**가 `dig_io.MoleculeShardReader`로 mmap 열고 `__getitem__`마다 `_record_to_data`로 **`Data`를 재구성**합니다.

정리: **shard는 `data.pt`를 대체하는 필수가 아니라, 같은 정보를 “한 파일 mmap”으로 읽기 위한 옵션**이고, `.pt`는 그대로 두고 왔다 갔다 할 수 있게 되어 있습니다 (`MoleculeNetShard.py` 주석 참고).

---

## 5. 뷰 커널·스캐폴드 Rust (shard와 별개)

- **뷰 함수들** (`uniform_sample_subgraph`, `rw_sample_subgraph`, `edge_perturb` 등):  
  - 그래프 `edge_index`를 넣고 **seed 고정**으로 서브그래프/perturb를 만듭니다.  
  - GIL 밖에서 돌릴 수 있게 설계된 **CPU/rayon 계열 마이크로 커널**입니다 (대조학습에서 반복 호출 비용을 줄이기 위함).

- **`key_split(..., impl='rust')`** (`dig/threedgraph/dataset/dataset.py`):  
  - Python이 `torch.randperm` 등으로 **키 순열**을 만든 뒤,  
  - **“안정 정렬 + 버킷 경계 반올림”**만 `dig_io.scaffold_bucket_sort`에 넘깁니다.  
  - 주석대로 **같은 시드면 기존 Torch 구현과 비트 단위로 맞출 목적**입니다.

이 부분은 **“그래프 텐서를 파일에 저장”하는 shard와는 다른 축**입니다.

---

## 6. 빌드·가용성·폴백

- 빌드: 레포 루트에서 **`cd dig_io && maturin develop --release`** (또는 `cargo build --release` + Windows DLL 폴백 — `dig_io/python/dig_io/__init__.py`에 `qdf_io`와 동일 패턴).  
- **`dig_io.is_available()`이 False**면:  
  - shard 쓰기/읽기는 에러 또는 “빌드하라” 메시지,  
  - `key_split`은 **`impl='python'`**으로 원래 Torch 경로.  
- 즉 **Rust 없이도 DGCL은 동작**하고, Rust는 **가속·I/O 경로 opt-in**입니다.

---

## 7. 한 문장 요약

- **`dig_io`의 shard** = **MoleculeNet `Data`를 `DIGSHRD` 단일 파일에 패킹 + mmap으로 랜덤 액세스**하는 DGCL 전용 포맷.  
- **QDF `qdf_io` shard** = **QDF 필드 전처리 레코드**용 **다른 포맷(`QDFSHRD`)**.  
- 그 외 Rust는 **대조 뷰 샘플링**과 **스캐폴드 분할 정렬**을 **Python과 동일 결과**를 목표로 보조합니다.

---

## 관련 경로

| 항목 | 경로 |
|------|------|
| Rust 소스·포맷 주석 | `dig_io/src/lib.rs` |
| Python 바인딩·가용성 | `dig_io/python/dig_io/__init__.py` |
| Shard 데이터셋 | `dig/threedgraph/dataset/MoleculeNetShard.py` |
| 스캐폴드 split Rust 위임 | `dig/threedgraph/dataset/dataset.py` (`key_split`, `_try_dig_io`) |
| 변환 스크립트 | `examples/sslgraph/convert_dataset_to_shard.py` |
| QDF ↔ DGCL 파이프라인 개요 | `docs/qdf_to_dgcl_pipeline.md`, `docs/qdf_to_dgcl_pipeline_two_tier.md` |
