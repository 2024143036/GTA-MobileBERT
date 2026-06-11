import pandas as pd
import re

# 1. 수집된 6만 건 통합 원천 데이터 로드
raw_file = "data/gta_all_raw_60k.csv"
try:
    df = pd.read_csv(raw_file)
    print(f"✅ 원본 데이터 로드 완료: 총 {len(df):,}건")
except FileNotFoundError:
    print(f"❌ '{raw_file}' 파일이 없습니다. 수집 완료 파일을 확인해 주세요.")
    exit()


# 2. 영문 텍스트 소문자화 및 정제
def clean_english_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # MobileBERT 학습용 소문자 변환
    text = re.sub(r"http\S+", "", text)  # 링크 제거
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)  # 특수문자 정제
    return re.sub(r"\s+", " ", text).strip()


print("⚙️ 텍스트 정제 중...")
df['clean_review'] = df['review'].apply(clean_english_text)

# 3. 영문 중립 키워드 기반 규칙 정의 (0:부정, 1:중립, 2:긍정)
neutral_keywords = [
    'average', 'mediocre', 'ok ', 'okay', 'decent', 'not bad',
    'but ', 'however', 'so-so', 'pros and cons', 'nothing special'
]


def assign_label(row):
    text = row['clean_review']

    # 문장 안에 중립 키워드가 하나라도 있으면 '중립(1)' 지정
    if any(kw in text for kw in neutral_keywords):
        return 1

        # 중립 키워드가 없고 스팀 추천(True)이면 '긍정(2)', 비추천(False)이면 '부정(0)'
    return 2 if row['voted_up'] == True else 0


print("🏷️ 6만 건 전체 데이터에 라벨(0, 1, 2) 부착 중...")
df['label'] = df.apply(assign_label, axis=1)

# 4. 데이터 분할 없이 '라벨링이 완료된 전체 데이터'를 새 파일로 저장
output_file = "data/gta_all_labeled_60k.csv"
df[['game', 'clean_review', 'label']].to_csv(output_file, index=False, encoding='utf-8-sig')

print("\n==================================================")
print(f"🏁 순수 라벨링 작업 완료!")
print(f"💾 최종 생성 파일: {output_file}")
print("==================================================")

# 5. 결과 확인을 위한 라벨별 데이터 분포 출력
print("\n📊 [최종 라벨링 데이터 분포 결과]")
print(df['label'].value_counts().rename({0: '부정(0)', 1: '중립(1)', 2: '긍정(2)'}))