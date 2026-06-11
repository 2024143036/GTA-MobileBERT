import requests
import pandas as pd
import time


def fetch_steam_reviews(app_id, game_name, target_count=30000):
    """
    스팀 API를 이용해 특정 GTA 시리즈의 영문 리뷰를 최대 30,000개 수집하는 함수
    """
    reviews_list = []
    cursor = '*'  # 첫 페이지 커서 초기화
    url = f"https://store.steampowered.com/appreviews/{app_id}"

    print(f"==================================================")
    print(f"[{game_name} (AppID: {app_id})] 목표: {target_count}개 영문 리뷰 수집 시작...")
    print(f"==================================================")

    while len(reviews_list) < target_count:
        params = {
            'json': 1,
            'filter': 'all',  # 최신/유용 리뷰 전체 대상
            'language': 'english',  # 대량 수집을 위해 영어 설정
            'cursor': cursor,  # 다음 페이지 이동을 위한 포인터
            'num_per_page': 100,  # 한 페이지당 최대 호출 수
            'purchase_type': 'all'  # 스팀 구매자 + 외부 키 포함
        }

        try:
            # 대량 수집이므로 안정성을 위해 타임아웃을 15초로 설정
            response = requests.get(url, params=params, timeout=15).json()
        except Exception as e:
            print(f"⚠️ 연결 오류 발생, 5초 대기 후 다시 시도합니다... ({e})")
            time.sleep(5)
            continue

        if response.get('success') != 1:
            print("❌ 스팀 API 호출 제한 또는 오류. 현재까지의 데이터만 저장하고 중단합니다.")
            break

        reviews = response.get('reviews', [])
        if not reviews:
            print(f"💡 안내: 해당 게임의 모든 영문 리뷰를 수집했습니다.")
            break

        for rev in reviews:
            clean_text = rev['review'].strip()
            if clean_text:  # 내용이 있는 실제 리뷰만 수집
                reviews_list.append({
                    'game': game_name,
                    'review': clean_text,
                    'voted_up': rev['voted_up'],  # True(긍정) / False(부정)
                    'playtime_forever': rev['author']['playtime_forever']  # 플레이 시간(분 단위)
                })

            if len(reviews_list) >= target_count:
                break

        print(f"-> 진행 상황: {len(reviews_list)} / {target_count} 건 수집 완료")

        # 다음 페이지 커서 갱신
        new_cursor = response.get('cursor')
        if cursor == new_cursor or not new_cursor:
            break
        cursor = new_cursor

        # 🔥 중요: 연속 호출로 인한 스팀 서버 차단(IP 블록) 방지용 딜레이 (0.6초)
        time.sleep(0.6)

    if reviews_list:
        df = pd.DataFrame(reviews_list)
        file_name = f"gta_{game_name.lower()}_30k_en_raw.csv"
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        print(f"✨ [{game_name}] 파일 저장 완료: {file_name} (최종 수집: {len(df)}건)\n")
        return df
    else:
        print(f"❌ [{game_name}] 수집된 데이터가 없습니다.\n")
        return None


# --- 실행부: GTA 5 및 GTA 4 각각 30,000건 세팅 ---
gta_series = {
    "GTA5": {"app_id": "271590", "target": 30000},
    "GTA4": {"app_id": "12210", "target": 30000}
}

combined_dfs = []
for game_name, info in gta_series.items():
    df = fetch_steam_reviews(info["app_id"], game_name, target_count=info["target"])
    if df is not None:
        combined_dfs.append(df)

# 모든 파일 하나로 통합하여 저장 (총합 60,000건 규모)
if combined_dfs:
    total_df = pd.concat(combined_dfs, ignore_index=True)
    total_df.to_csv("gta_all_raw_60k.csv", index=False, encoding='utf-8-sig')
    print(f"🏁 [전체 완료] 6만 건 통합본 파일 생성 완료: gta_all_raw_60k.csv (총 데이터 양: {len(total_df)}건)")