# Mermaid → SVG / PNG

소스는 `docs/qdf_to_dgcl_pipeline.md`, `docs/qdf_to_dgcl_pipeline_two_tier.md` 안의 fenced `mermaid` 코드 블록입니다.  
추출본은 `src/*.mmd` 에 덮어씁니다.

## 생성

레포 루트에서:

```bash
.venv/bin/python docs/figures/render_mermaid_diagrams.py
```

- 기본 **Kroki** (`kroki.io`, HTTPS, 인터넷 필요). Python 기본 UA는 403이 나올 수 있어 스크립트에 브라우저형 `User-Agent`를 넣었습니다.
- 큰 도표의 **PNG**가 Kroki에서 실패하면, `rsvg-convert`가 있으면 방금 쓴 **SVG → PNG**로 대체합니다 (`apt install librsvg2-bin`).
- 오프라인: `--renderer mmdc` — 로컬 `mmdc` 또는 `npx @mermaid-js/mermaid-cli` (Chromium/Puppeteer 필요). 샌드박스용 예시는 `docs/figures/puppeteer-config.json` 입니다.

## 산출물

| stem | 설명 |
|------|------|
| `qdf_to_dgcl_pipeline` | 통합 한 장 플로차트 |
| `qdf_to_dgcl_two_tier_data` | 2단 문서 — 데이터 평면 |
| `qdf_to_dgcl_two_tier_code` | 2단 문서 — 코드 평면 |

Kroki용 Mermaid는 노드 문자열 안의 **백틱(`)** 이 렉서 오류를 낼 수 있어, 위 md에는 백틱 없이 경로만 적어 두었습니다.
