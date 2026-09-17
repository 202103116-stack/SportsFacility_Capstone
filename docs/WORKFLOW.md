# ML Workflow — 전국 체육시설 안전위험 예측 프로젝트

LG Aimers 9기 해커톤에서 쓰던 9단계 ML 워크플로우를 이 프로젝트에 맞게 적용한 버전.
각 단계가 이 프로젝트의 어떤 파일/폴더에 대응하는지 명시한다.

## 현재 진행 상태

| 단계 | 상태 |
|---|---|
| 1. Problem definition | ✅ 완료 (`CLAUDE.md`) — 단, 타겟 재정의 반영 필요 |
| 2. Data acquisition & understanding | ✅ 완료 (`docs/data_sources.md`, `docs/data_dictionary.md`) |
| 3. Data quality / leakage audit | ✅ 1차 완료 (`docs/eda_findings.md`) — ⚠️ 아래 3번 설계 이슈 확인 필요 |
| 4. Validation design | 🔲 다음 단계 |
| 5. Baseline + preprocessing/FE | 🔲 |
| 6. Modeling & training | 🔲 |
| 7. Evaluation & error analysis | 🔲 |
| 8. Final training & packaging | 🔲 |
| 9. Monitoring & iteration | 🔲 |

---

## 1. Problem Definition

**목표**: 시설 특성(업종/시설유형/지역/면적/시설연령/점검경과일)으로 안전위험군을 예측하는
분류 모델 → 예측 위험도 기반 점검자원 우선순위 로직 → 사고통계 교차검증.

**타겟 (EDA로 확정, CLAUDE.md 수정 반영)**:
- ~~3-클래스(주의/사용중지/수리필요)~~ → **이진분류: 양호(0) vs 위험군=주의+사용중지(1)**
  (근거: `사용중지`가 전국 23건뿐이라 별도 클래스 학습 불가 — `docs/eda_findings.md` §2)
- 클래스 비율: 위험군 3.85% → 심각한 불균형, `class_weight`/리샘플링 필수
- 분석 대상: `faci_stat_nm == '정상운영'`으로 한정 권장(폐업 시설은 점검자원 배분 대상 아님)
- 라벨 결측(정상운영 기준 23.3%)은 학습에서 제외

**성공 지표**: Accuracy 지양, Recall/Precision/F1/PR-AUC 우선 (CLAUDE.md 명시). 불균형
이진분류이므로 특히 PR-AUC와 Recall(위험 시설을 놓치지 않는 것)을 핵심 지표로 삼는다.

---

## 2. Data Acquisition & Understanding

- 소스: `docs/data_sources.md` (오픈API 3종 + 보조 데이터)
- 필드 사전: `docs/data_dictionary.md` (33개 컬럼 전체, 공식 스펙 기반)
- 수집 스크립트: `src/fetch_facility_data.py` (safety/schk/atnm/facility 대상 선택 가능)
- 현재 스냅샷: `data/raw/facility_safety_20260917.csv` (98,551행, 전수)

---

## 3. Data Quality / Leakage Audit

`docs/eda_findings.md`, `docs/data_dictionary.md` §4에 정리된 품질 이슈(면적 이상치,
날짜 더미값, 의미불명 코드필드) 외에 **구조적으로 중요한 설계 이슈**가 하나 있다:

> ⚠️ **`TODZ_API_FACI_SAFETY`는 시설당 "가장 최근 점검 결과" 1행만 제공한다.**
> 즉 `schk_visit_ymd`(점검방문일)는 현재 라벨(`schk_tot_grd_nm`)을 만든 바로 그 점검의
> 날짜다. 이걸로 만든 `days_since_inspection`은 "예전 점검 이후 지금까지 얼마나
> 지났나"가 아니라 "그 라벨을 만든 점검이 언제 있었나"에 가깝다 — CLAUDE.md가 원래
> 의도한 "마지막 점검 후 경과일수로 위험도를 예측한다"는 스토리와 미묘하게 어긋난다.

**두 가지 옵션 중 하나를 택해야 함(다음 세션에서 결정):**

- **Option A (현재 범위, 권장 — 시간 제약 고려)**: `TODZ_API_FACI_SAFETY` 단독으로
  "시설의 현재 속성이 현재 위험등급과 어떤 연관이 있는가"를 보는 **연관성/설명 모델**로
  프레이밍한다. 인과적으로 "미래 위험을 예측"한다고 과장하지 않고, "이런 속성의 시설이
  위험군일 가능성이 높다 → 우선 점검 대상으로 스코어링한다"는 정도로 명확히 범위를 좁힌다.
- **Option B (더 엄밀함, 데이터 작업 추가 필요)**: `TODZ_API_FACI_SCHK`(점검 이력,
  267,694건)로 시설별 점검 이력을 시계열로 재구성 → "T-1 시점 속성으로 T 시점 등급을
  예측"하는 패널 구조로 바꾸면 진짜 "예측" 모델이 된다. 데이터 엔지니어링 비용이 큼.

이 캡스톤 일정상 **Option A로 진행하고 발표에서도 그렇게 정직하게 설명하는 것을 권장**한다.
(제조/품질관리 유비 — "현재 설비 속성과 현재 불량 여부의 연관성 분석"도 동일 논리로 유효함)

---

## 4. Validation Design (다음 단계)

- Train/Test 분할: `risk_binary` 기준 **stratified split** (불균형 유지 확인용)
- 교차검증: Stratified K-Fold (k=5), 폴드마다 위험군 비율 유지
- 리키지 주의: `schk_tot_text_cn`(안전점검종합평가내용), `schk_refm_text_cn`(개선방향내용)은
  점검 "결과"이자 라벨과 사실상 동의어이므로 **피처로 절대 사용 금지**
- 지표: PR-AUC(주지표), Recall, Precision, F1 — Accuracy는 참고용으로만 표기

---

## 5. Baseline + Preprocessing / Feature Engineering (다음 단계)

- 전처리: `docs/data_dictionary.md` §4 품질 이슈 그대로 적용
  (면적 클리핑, 날짜 더미값 제거, 정상운영 필터, 라벨 결측 제외)
- 피처: `fcob_nm`/`ftype_nm`/`cp_nm`(범주형 인코딩), `area_sqm`, `facility_age_years`,
  `days_since_inspection`(Option A 전제하 사용), `faci_gb_nm`, `atnm_chk_yn`
- 베이스라인: 로지스틱 회귀(`class_weight='balanced'`) → 성능 하한선 확보
- 예정 파일: `src/preprocess.py`, `src/features.py`

---

## 6. Modeling & Training (다음 단계)

- 트리 기반 앙상블(RandomForest → LightGBM/XGBoost) 순서로 고도화
- 불균형 대응: `class_weight`, `scale_pos_weight`, 또는 SMOTE류 오버샘플링 비교
- 예정 파일: `src/train.py`, `models/`에 저장

## 7. Evaluation & Error Analysis (다음 단계)

- Confusion Matrix, PR curve, Feature Importance/SHAP
- 오탐(정상인데 위험 예측)·미탐(위험인데 정상 예측) 케이스의 업종/지역 패턴 분석
- 필요시 5~6단계로 회귀(피처 추가, 임계값 조정)

## 8. Final Training & Packaging (다음 단계)

- 전체 데이터로 최종 재학습 → `models/`에 저장
- 자원배분 우선순위 로직(2단계 최적화)과 연결: 위험확률 × 시설 중요도(면적/이용규모 등
  가중치) 스코어링 → 예산/인력 제약 하 상위 N개 선정
- 산출물: `outputs/reports/`

## 9. Monitoring & Iteration (다음 단계)

- 이번 캡스톤 맥락에서는 "배포 후 모니터링" 대신, 스포츠안전사고 통계(3단계 검증)와
  교차검증한 결과를 기록하고 다음 가설(피처 추가, 임계값 재조정)을 결정하는 절차로 대체
