"""
전국체육시설 정보 / 안전점검 정보 오픈API 수집 스크립트

사용법:
    python src/fetch_facility_data.py --target safety      # 시설 마스터+현재 안전등급 (권장, ~98,551건)
    python src/fetch_facility_data.py --target schk        # 안전점검결과(종합) 이력 (~267,694건)
    python src/fetch_facility_data.py --target facility    # 전국체육시설 일반정보 (~153,494건)

.env 파일에 DATA_GO_KR_SERVICE_KEY, FACILITY_INFO_ENDPOINT, FACILITY_SAFETY_ENDPOINT가 설정되어 있어야 함.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY")
FACILITY_INFO_ENDPOINT = os.getenv("FACILITY_INFO_ENDPOINT")
FACILITY_SAFETY_ENDPOINT = os.getenv("FACILITY_SAFETY_ENDPOINT")

# target 이름 -> (base endpoint, 세부기능 operation, 한글 설명, 저장 파일명 접두)
TARGETS = {
    "facility": (FACILITY_INFO_ENDPOINT, "TODZ_API_SFMS_FACI", "전국체육시설 일반정보", "facility_info"),
    "safety": (FACILITY_SAFETY_ENDPOINT, "TODZ_API_FACI_SAFETY", "체육시설안전정보(시설당 1행, 권장)", "facility_safety"),
    "schk": (FACILITY_SAFETY_ENDPOINT, "TODZ_API_FACI_SCHK", "안전점검결과(종합) 이력", "safety_check_history"),
    "atnm": (FACILITY_SAFETY_ENDPOINT, "TODZ_API_FACI_ATNM", "자율안전점검결과(종합) 이력", "self_check_history"),
}

NUM_OF_ROWS = 1000  # 페이지당 요청 건수
SLEEP_SEC = 0.2      # 요청 간 간격 (공공API 매너용)


def fetch_page(base_endpoint: str, operation: str, page_no: int, num_of_rows: int = NUM_OF_ROWS) -> dict:
    url = f"{base_endpoint}/{operation}"
    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "resultType": "json",
    }
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    return res.json()


def extract_items(payload: dict) -> list:
    body = payload.get("body", {})
    items = body.get("items", [])
    # 일부 공공API는 items 안에 {"item": [...]} 형태로 한 겹 더 감싸져 있음 -> 방어적으로 처리
    if isinstance(items, dict) and "item" in items:
        items = items["item"]
        if isinstance(items, dict):
            items = [items]
    return items


def fetch_all(target: str) -> pd.DataFrame:
    base_endpoint, operation, desc, _ = TARGETS[target]
    if not base_endpoint:
        raise RuntimeError(f".env에 해당 endpoint가 비어있습니다: {target}")

    print(f"[{target}] {desc} 수집 시작 ({operation})")

    first = fetch_page(base_endpoint, operation, page_no=1, num_of_rows=1)
    result_code = first.get("header", {}).get("resultCode")
    if result_code != "00":
        raise RuntimeError(f"API 응답 오류: {first.get('header')}")

    total_count = first.get("body", {}).get("totalCount", 0)
    total_pages = (total_count + NUM_OF_ROWS - 1) // NUM_OF_ROWS
    print(f"  전체 건수(totalCount): {total_count:,}건 -> {total_pages}페이지 수집 예정")

    all_items = []
    for page_no in range(1, total_pages + 1):
        payload = fetch_page(base_endpoint, operation, page_no=page_no)
        items = extract_items(payload)
        all_items.extend(items)
        if page_no % 10 == 0 or page_no == total_pages:
            print(f"  진행: {page_no}/{total_pages} 페이지 ({len(all_items):,}건 누적)")
        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(all_items)
    print(f"[{target}] 수집 완료: 실제 {len(df):,}행 (API 신고 건수 {total_count:,}건)")
    return df


def main():
    parser = argparse.ArgumentParser(description="전국체육시설 오픈API 수집기")
    parser.add_argument(
        "--target",
        choices=list(TARGETS.keys()),
        default="safety",
        help="수집할 데이터 종류 (기본값: safety = 시설당 1행 안전정보, 권장)",
    )
    args = parser.parse_args()

    if not SERVICE_KEY:
        raise RuntimeError(".env에서 DATA_GO_KR_SERVICE_KEY를 찾을 수 없습니다. .env 파일을 확인하세요.")

    df = fetch_all(args.target)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, prefix = TARGETS[args.target]
    today = datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"{prefix}_{today}.csv"

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
