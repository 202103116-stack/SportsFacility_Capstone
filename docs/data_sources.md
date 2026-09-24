# 데이터 소스 정리

이번 학기 조사·검증한 데이터셋 전체 목록. (검증일 기준 2026-09)

## 핵심 데이터 (메인 모델링용)

### 1. 전국체육시설 안전점검 정보
- **오픈API**: 서울올림픽기념국민체육진흥공단_전국체육시설 안전점검 정보
  - data.go.kr: https://www.data.go.kr/data/15107773/openapi.do
  - Endpoint: `https://apis.data.go.kr/B551014/SRVC_API_FACI_SCHK_RESULT`
  - 세부기능:
    - `/TODZ_API_FACI_SAFETY` (체육시설안전정보 조회) — 시설 마스터 + 현재 안전등급, **totalCount 98,551건** (시설당 거의 1행)
    - `/TODZ_API_FACI_SCHK` (안전점검결과 종합 조회) — 점검 건별 이력 로그, **totalCount 267,694건**
    - `/TODZ_API_FACI_ATNM` (자율안전점검결과 종합 조회)
    - 분야별 조회 2종 (자율/일반)
  - 일일 트래픽: 기능별 각 10,000건
- **CSV(문화빅데이터플랫폼) — 필요시 별도 다운로드**: `KS_ALSFC_SAFECHK_INFO_202607.csv`
  - 메인 소스는 오픈API(`TODZ_API_FACI_SAFETY`)이며, 이 CSV는 필드 대조·교차검증용 참고 자료. 프로젝트에 아직 반입되지 않음.
  - https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=1762f1c0-2594-11eb-af9a-4b03f0a582d6
  - 192,314행 × 30컬럼. 오픈API와 필드 1:1 대응 확인됨(같은 원천).
  - ⚠️ 시설 마스터 + 이력이 섞여 있어 "시설명+주소" 기준 72,744개 그룹이 중복(전체의 약 40%). 전처리 필수.
  - ⚠️ 면적(FCLTY_TOTAR_CO) 컬럼에 날짜값 오입력 등 이상치 다수(10,000㎡ 초과 1,587건).
  - ⚠️ 날짜 필드(SAFECHK_DE, OPER_CLSBIZ_DE)에 오타/더미값(99991231 등) 존재.

**권장**: `TODZ_API_FACI_SAFETY`(98,551건, 시설당 1행)를 API로 새로 받아 `data/raw/`에 저장 → 기존 CSV보다 깨끗한 마스터 테이블로 활용. 이력 변수가 필요하면 `TODZ_API_FACI_SCHK`를 `faci_cd` 기준으로 조인.

### 2. 전국체육시설 정보 (시설 일반정보)
- data.go.kr: https://www.data.go.kr/data/15113986/openapi.do (오픈API, API명 TODZ_API_SFMS_FACI)
- data.go.kr: https://www.data.go.kr/data/15096288/standard.do (표준데이터 — 위와 동일 서비스로 확인됨, 활용신청 시 "이미 신청됨"으로 중복 차단)
- Endpoint: `https://apis.data.go.kr/B551014/SRVC_API_SFMS_FACI`
- **totalCount: 153,494건** (전국 체육시설 전체, 안전정보와 별개로 시설 상세정보)

## 보조/맥락 데이터

| 데이터셋 | 출처 | 역할 |
|---|---|---|
| KS_ALSFC_NEARBY_PBTRNSP_INFO_202607 (체육시설 인접 대중교통정보) | 문화빅데이터플랫폼 | 접근성 변수 (도보거리/이동시간 등) |
| KS_WNTY_PUBLIC_PHSTRN_FCLTY_STTUS_202607 (전국공공체육시설현황) | 문화빅데이터플랫폼 | 공공시설 한정 상세정보 |
| 국민체력100 측정결과 | 국민체육진흥공단·공공데이터포털 | 연령/성별 체력 정상범위 참고표 → 파생 프록시 변수용 |
| 스포츠안전사고 통계 | 스포츠안전재단 | 모델 결과 타당성 교차검증(도메인 근거) |
| 2025년 국민생활체육조사 | 문화체육관광부 | 지역별 생활체육 참여율 → 수요 강도 참고지표 |

## API 사용 시 참고

- 인증키(서비스키)와 두 엔드포인트(`DATA_GO_KR_SERVICE_KEY`, `FACILITY_INFO_ENDPOINT`, `FACILITY_SAFETY_ENDPOINT`)는 `.env` 파일로 관리하고 **절대 git에 커밋하지 않음** (`.gitignore`에 등록됨)
- 페이지네이션: `totalCount`를 먼저 확인 후 `pageNo`를 반복 호출해서 전체 수집 → `data/raw/`에 스냅샷 CSV로 저장 → 이후 분석은 이 스냅샷 기준으로 진행(재현성 확보)
- 공식 갱신주기는 데이터셋 페이지에 명시되어 있지 않지만, 실제 데이터는 거의 매일 갱신된다(수집 하루 전 날짜의 수정 기록이 존재). 그래서 한 시점의 스냅샷으로 고정해서 쓴다.
- 응답 JSON은 `{"response": {"header": ..., "body": ...}}`로 한 겹 감싸져 있다(`src/fetch_facility_data.py`가 처리).

## 수집한 스냅샷 (`data/raw/`, git 미추적)

| 파일 | 원천 | 행 수 | 용도 |
|---|---|---|---|
| `facility_safety_20260917.csv` | `TODZ_API_FACI_SAFETY` | 98,551 | **분석 기준 스냅샷(고정)**. `src/preprocess.py`·`src/eda.py`가 이 파일을 읽음 |
| `facility_safety_20260924.csv` | 〃 | 98,640 | 재수집 비교용(신규 시설 91개, 라벨 전환 0건 확인). 분석에는 쓰지 않음 |
| `safety_check_history_20260917.csv` | `TODZ_API_FACI_SCHK`(공식점검 이력) | 267,694 | Option B(보류) 및 라벨 출처 확인용 |
| `self_check_history_20260925.csv` | `TODZ_API_FACI_ATNM`(자율점검 이력) | 480,443 | 〃. 기록의 46.2%는 등급 공란 |

SAFETY 등급의 출처(공식점검 vs 자율점검)와 이력 커버리지는 `docs/data_dictionary.md` §6, 의사결정 근거는 `docs/decision_log.md` 참고.
