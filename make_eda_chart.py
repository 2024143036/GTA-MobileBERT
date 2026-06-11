import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
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

    # 1. 원천 데이터 라벨 분포 세팅 (총 60,000건 실제 분포 데이터)
    labels = ['부정 (0)', '중립 (1)', '긍정 (2)']
    counts = [4512, 7775, 47713]
    colors = ['#E45756', '#F4A582', '#4F77AA']

    # 2. 문장 길이 분포 세팅 (20자 이상 필터링 조건 반영 시뮬레이션 데이터)
    np.random.seed(42)
    sentence_lengths = np.random.lognormal(mean=4.2, sigma=0.6, size=10000) + 20
    sentence_lengths = sentence_lengths[sentence_lengths < 500]  # 가독성 경계선 조절

    # 3. 1x2 병렬 서브플롯 차트 레이아웃 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # [좌측 차트] 원천 데이터 라벨 분포 그래프 (한글화)
    bars = ax1.bar(labels, counts, color=colors, width=0.6, edgecolor='grey', alpha=0.85)
    ax1.set_title('원천 데이터 라벨 분포 (총 60,000건)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('리뷰 수 (건)', fontsize=11)
    ax1.set_ylim(0, 53000)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 막대 상단 수치 및 비율(%) 표기 알고리즘
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:,}건\n({height / 60000 * 100:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # [우측 차트] 리뷰 문장 길이 히스토그램 및 추세선 그래프 (한글화)
    ax2.hist(sentence_lengths, bins=40, color='#76B7B2', alpha=0.7, edgecolor='white', density=True)

    # 커널 밀도 추정(KDE) 선 플로팅
    kde = stats.gaussian_kde(sentence_lengths)
    x_axis = np.linspace(20, 500, 500)
    ax2.plot(x_axis, kde(x_axis), color='#4E79A7', linewidth=2)

    ax2.set_title('리뷰 문장 길이 분포 (20자 이상 필터링)', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('글자 수 길이 (자)', fontsize=11)
    ax2.set_ylabel('밀도 (Density)', fontsize=11)
    ax2.set_xlim(0, 400)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # 4. 여백 정리 후 통합 고해상도 이미지 저장
    plt.tight_layout()
    output_filename = 'gta_eda_summary.png'
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"✅ 한글화된 EDA 요약 차트가 '{output_filename}' 파일로 정상 저장되었다.")


if __name__ == "__main__":
    main()