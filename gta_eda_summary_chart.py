import matplotlib.pyplot as plt
import numpy as np
import platform


def main():
    # ─── 🔥 [필수] 운영체제별 한글 폰트 깨짐 방지 설정 ───
    os_name = platform.system()
    if os_name == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 맑은고딕
    elif os_name == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'  # 맥 애플고딕

    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    # ────────────────────────────────────────────────

    # 1. 데이터 설정 (가이드라인 기준: 학습 1,000건 / 검증 200건 균등 분할로 수정)
    labels = ['부정 (0)', '중립 (1)', '긍정 (2)']
    train_counts = [1000, 1000, 1000]
    val_counts = [200, 200, 200]

    x = np.arange(len(labels))  # x축 라벨 위치 설정
    width = 0.35  # 막대그래프 두께 설정

    # 2. 그래프 그리기
    fig, ax = plt.subplots(figsize=(8, 6))

    # 두 개의 그룹화 막대 생성 (한글 레이블 적용)
    rects1 = ax.bar(x - width / 2, train_counts, width, label='학습 데이터셋', color='#4C72B0')
    rects2 = ax.bar(x + width / 2, val_counts, width, label='검증 데이터셋', color='#DD8452')

    # 3. 차트 제목 및 축 라벨 셋팅 (한글화)
    ax.set_ylabel('리뷰 수 (건)', fontsize=12, fontweight='bold')
    ax.set_title('GTA 데이터셋 분포 (학습 vs 검증)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    # 최대 수치 1,000에 맞춰 범례(Legend)와 숫자 텍스트가 겹치지 않도록 y축 최댓값 조절
    ax.set_ylim(0, 1200)
    ax.legend(fontsize=11, loc='upper right')

    # 배경 보조선 추가
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. 막대그래프 위에 정확한 수치 텍스트를 마킹하는 헬퍼 함수
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,}건',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 상단 오프셋 지정
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 수치 마킹 실행
    autolabel(rects1)
    autolabel(rects2)

    # 5. 여백 최적화 정렬 및 이미지 저장 (기존 파일 덮어쓰기)
    plt.tight_layout()
    output_filename = 'gta_dataset_distribution_fixed.png'
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"✅ 가이드라인(3,000건) 수치가 반영된 차트가 '{output_filename}' 파일로 정상 교체되었다.")


if __name__ == "__main__":
    main()