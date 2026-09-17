# SportsFacility_Capstone

한국외국어대학교 26-2학기 「스포츠융합산업프로젝트실습」 캡스톤 프로젝트

## 프로젝트 개요

- **주제 방향**: 전국 체육시설 안전점검 데이터를 활용한 시설 안전위험 예측 및 점검자원 우선순위 최적화
- **핵심 흐름**: (1) 시설 특성 기반 안전위험군 예측(분류 모델) → (2) 예측 결과 기반 점검자원 우선순위 제안(운영 최적화) → (3) 스포츠안전사고 통계로 타당성 교차검증
- **의의**: 스포츠 도메인에 한정되지 않고 제조/생산 현장의 예지보전(predictive maintenance)·검사 우선순위 결정 문제와 구조가 동일 — AI 활용 데이터분석 역량 연습용으로도 활용

## 폴더 구조

```
SportsFacility_Capstone/
├── data/
│   ├── raw/          # 원본 데이터 (API pull, CSV 다운로드본 그대로)
│   ├── processed/     # 전처리 완료된 데이터 (중복제거, 결측치 처리 등)
│   └── external/      # 보조 데이터 (국민체력100, 스포츠안전사고 통계 등)
├── notebooks/          # EDA·모델링 주피터 노트북
├── src/                # 재사용 스크립트 (API 수집, 전처리, 모델링 함수)
├── models/             # 학습된 모델 저장 (.pkl 등)
├── outputs/
│   ├── figures/        # 시각화 결과물
│   └── reports/        # 보고서, 발표자료 산출물
└── docs/               # 데이터 출처, 회의록, 참고문헌, 계획서
```

## 데이터 소스

`docs/data_sources.md` 참고 — 지금까지 조사·검증한 모든 데이터셋/오픈API 목록과 검증 결과가 정리되어 있음.
컬럼 단위 상세 정의·실측 통계·데이터 품질 이슈는 `docs/data_dictionary.md` 참고.

## 환경 설정

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # 이후 .env에 발급받은 인증키 직접 입력
python src\fetch_facility_data.py --target safety
```

- 시스템에 Python 버전이 여러 개 있다면 `.venv`가 없는 기본 `python`/`py`는 패키지가
  없을 수 있으니, 반드시 `.venv` 활성화 후 작업할 것.
- `.env`는 절대 git에 커밋하지 않음(`.gitignore`에 등록됨). 키를 다른 사람에게
  공유할 때도 `.env` 파일 자체가 아니라 안전한 채널로 전달할 것.

## 작업 환경

- VS Code + Claude Code로 이 폴더를 열어서 작업
- Claude 데스크 앱(Cowork) 세션에도 이 폴더가 연결되어 있어 클라우드 쪽에서도 동일 폴더 기준으로 작업 가능
