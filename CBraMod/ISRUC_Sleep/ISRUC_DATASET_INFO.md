# ISRUC-Sleep Dataset - 전처리 정보 총정리

## 📚 목차
1. [데이터셋 개요](#데이터셋-개요)
2. [원본 데이터 구조](#원본-데이터-구조)
3. [전처리 파이프라인](#전처리-파이프라인)
4. [라벨링 및 클래스 불균형](#라벨링-및-클래스-불균형)
5. [원본 vs DIVER 적용 비교표](#원본-vs-diver-적용-비교표)
6. [구현 상세](#구현-상세)

---

## 데이터셋 개요

### 기본 정보
- **데이터셋명**: ISRUC-Sleep (Instituto Superior de Engenharia do Porto - Sleep Dataset)
- **출처**: ISEP (Instituto Superior de Engenharia do Porto), Portugal
- **목적**: 수면 단계 분류 (Sleep Stage Classification)
- **대상**: 수면 장애 환자 및 정상인
- **데이터 타입**: 두피 뇌파 (Scalp EEG) + 다중 생체신호 (PSG)

### 서브젝트 정보
- **전체 서브젝트 수**: 100명 (Subgroup 1)
- **서브젝트 번호**: Subject 1 ~ 100
- **수면 기록**: 각 서브젝트당 1-2개 밤 기록
- **나이 분포**: 20-85세
- **성별**: 남성/여성 포함

---

## 원본 데이터 구조

### 파일 구조
```
/ISRUC_S1/
├── 1/
│   ├── 1.rec             # PSG 데이터 (다중 채널)
│   ├── 1_1.txt           # 수면 단계 라벨 (30초 epoch)
│   └── ...
├── 2/
└── ...
```

### 채널 정보

#### 전극 시스템
- **시스템**: 10-20 International System
- **채널 수**: 6개 EEG 채널 (표준 수면 연구 채널)
- **Reference**: Contralateral mastoid (A1, A2)

#### 6개 EEG 채널
```python
eeg_channels = [
    "F3-A2",  # Left Frontal
    "C3-A2",  # Left Central
    "F4-A1",  # Right Frontal
    "C4-A1",  # Right Central
    "O1-A2",  # Left Occipital
    "O2-A1"   # Right Occipital
]
```

#### 기타 생체신호 (PSG)
- **EOG** (Electrooculography): 안구 움직임
- **EMG** (Electromyography): 턱 근전도
- **ECG** (Electrocardiography): 심전도

> **DIVER 전처리에서는 EEG 6개 채널만 사용**

### Recording 정보
- **Sampling Rate**: 200 Hz
- **파일 형식**: `.rec` (바이너리) + `.txt` (라벨)
- **Recording 길이**: 전체 수면 시간 (약 6-8시간)
- **Epoch 길이**: 30초 (수면 연구 표준)
- **데이터 타입**: Float (μV 단위)

---

## 전처리 파이프라인

### 파이프라인 구조
```
Raw .rec Files → EEG 채널 추출 → 30초 Epoch 분할 → 라벨 매칭 → LMDB
```

### Stage 1: 원시 데이터 로드

#### 목적
- `.rec` 파일에서 EEG 6개 채널만 추출
- PSG의 다른 채널 제외
- 200 Hz sampling rate 확인

#### 처리 과정
1. **`.rec` 파일 파싱**
   - 바이너리 파일 읽기
   - 채널 메타데이터 확인
   - EEG 채널만 선택

2. **채널 정렬**
   - F3-A2, C3-A2, F4-A1, C4-A1, O1-A2, O2-A1 순서
   - 누락된 채널 처리

### Stage 2: Epoch 분할 및 라벨 매칭

#### Epoch 정의
- **Epoch 길이**: 30초 (수면 연구 표준)
- **Sampling rate**: 200 Hz
- **Epoch 샘플 수**: 30초 × 200Hz = 6000 샘플

#### 라벨 파일 구조
```
# 1_1.txt
0     # Epoch 1 → Wake (W)
0     # Epoch 2 → Wake (W)
1     # Epoch 3 → N1
2     # Epoch 4 → N2
3     # Epoch 5 → N3
5     # Epoch 6 → REM
...
```

#### 라벨 매핑
```python
label_mapping = {
    0: "W",    # Wake (각성)
    1: "N1",   # NREM Stage 1
    2: "N2",   # NREM Stage 2
    3: "N3",   # NREM Stage 3 (Deep Sleep)
    5: "REM"   # REM Sleep
}

# DIVER 학습용 5-class
# 0: W, 1: N1, 2: N2, 3: N3, 4: REM (5→4로 변환)
```

### Stage 3: Train/Val/Test Split

#### 분할 전략 (CBraMod 논문 기준)
- **Train**: Subject 1 ~ 84
- **Validation**: Subject 85 ~ 90 (6명)
- **Test**: Subject 91 ~ 100 (10명)

```python
train_subjects = list(range(1, 85))   # 84명
val_subjects = list(range(85, 91))    # 6명
test_subjects = list(range(91, 101))  # 10명
```

> **Subject-level split**: Data leakage 방지

---

## 라벨링 및 클래스 불균형

### 라벨 정의 (5-class)
- **Label 0 (W)**: Wake (각성 상태)
- **Label 1 (N1)**: NREM Stage 1 (얕은 수면)
- **Label 2 (N2)**: NREM Stage 2 (중간 수면)
- **Label 3 (N3)**: NREM Stage 3 (깊은 수면, Slow-Wave Sleep)
- **Label 4 (REM)**: REM Sleep (렘수면)

### 수면 단계별 특징

| Stage | 이름 | 뇌파 특징 | 비율 (%) |
|-------|------|----------|---------|
| **W** | Wake | Beta waves (고주파) | ~5% |
| **N1** | Light Sleep | Theta waves | ~5% |
| **N2** | Moderate Sleep | Sleep spindles, K-complexes | ~50% |
| **N3** | Deep Sleep | Delta waves (저주파) | ~20% |
| **REM** | REM Sleep | Mixed frequency, low amplitude | ~20% |

### 클래스 불균형

#### 전형적인 수면 구조
```
전체 수면 시간: ~480 epochs (8시간)
- W: 24 epochs (~5%)
- N1: 24 epochs (~5%)
- N2: 240 epochs (~50%)  ← 매우 많음!
- N3: 96 epochs (~20%)
- REM: 96 epochs (~20%)
```

#### 불균형 해결 방법
1. **Class weights**: N2에 낮은 weight, N1에 높은 weight
2. **Oversampling**: 적은 클래스(W, N1) 증강
3. **Weighted loss**: Cross-entropy에 class weight 적용

---

## 원본 vs DIVER 적용 비교표

| 구분 | 항목 | 원본 (ISRUC Original) | 수정 (DIVER 적용) |
|------|------|---------------------|-------------------|
| **데이터셋** | 전체 서브젝트 수 | 100명 (Subgroup 1) | **100명** (동일) |
| | Train | 1 ~ 84 (84명) | **84명** (동일) |
| | Validation | 85 ~ 90 (6명) | **6명** (동일) |
| | Test | 91 ~ 100 (10명) | **10명** (동일) |
| **원본 데이터** | Sampling Rate | 200 Hz | **200 Hz** (동일) |
| | 파일 형식 | `.rec` (바이너리) | **`.rec`** (동일) |
| | 사용 채널 | EEG 6개 | **EEG 6개** (동일) |
| **채널 시스템** | 전극 배치 시스템 | 10-20 System | **10-20 System** (동일) |
| | Reference | Mastoid (A1, A2) | **Mastoid** (동일) |
| | 채널 목록 | F3-A2, C3-A2, F4-A1, C4-A1, O1-A2, O2-A1 | **동일** |
| | ELC 파일 사용 | ❌ 없음 | ✅ **사용** (standard_1005.elc) |
| **전처리** | Epoch 길이 | 30초 | **30초** (동일) |
| | Epoch 샘플 수 | 6000 (30초 × 200Hz) | **6000** (동일) |
| | 슬라이딩 윈도우 | Non-overlapping (30초 step) | **Non-overlapping** (동일) |
| **리샘플링** | 타겟 Sampling Rate | - (200 Hz 유지) | ✅ **500 Hz** |
| | 리샘플링 방법 | - | **scipy.signal.resample** |
| | Epoch 샘플 수 변경 | 6000 → | **15000** (30초 × 500Hz) |
| | Reshape | - | ✅ **(6, 15000) → (6, 30, 500)** |
| **라벨링** | 라벨 종류 | 5-class (W, N1, N2, N3, REM) | **5-class** (동일) |
| | Label 0 | W (Wake) | **W** (동일) |
| | Label 1 | N1 (NREM 1) | **N1** (동일) |
| | Label 2 | N2 (NREM 2) | **N2** (동일) |
| | Label 3 | N3 (NREM 3) | **N3** (동일) |
| | Label 4 | REM (원래 5) | **REM** (5→4 변환) |
| | 라벨 출처 | `{subject}_1.txt` 파일 | **동일** |
| **최종 출력** | Shape | **(6, 6000)** | ✅ **(6, 30, 500)** |
| | | 6채널, 6000샘플 | 6채널, 30×1초, 500샘플/초 |
| | 데이터 구조 | `{"signal": array, "label": int}` | `{"signal": array, "label": int, "elc_info": dict}` |
| | 저장 형식 | `.npy` or `.h5` | ✅ **LMDB** |
| **정규화** | Z-score 정규화 | ❌ 없음 | ❌ **없음** (모델에서 처리) |

---

## 구현 상세

### Shape 변환 과정
```python
# 원본 (200Hz, 30초 epoch)
signal_200 = np.array (6, 6000)  # 6채널 × 30초 × 200Hz

# Step 1: 리샘플링 (200Hz → 500Hz)
from scipy.signal import resample
signal_500 = resample(signal_200, 15000, axis=1)  # (6, 15000)

# Step 2: Reshape (30개 1초 세그먼트)
signal_final = signal_500.reshape(6, 30, 500)  # (6, 30, 500)
```

### ELC 파일 구조
```python
elc_info = {
    "channel_names": [
        "F3-A2", "C3-A2",
        "F4-A1", "C4-A1",
        "O1-A2", "O2-A1"
    ],
    "electrode_pairs": {
        "F3-A2": ["F3", "A2"],
        "C3-A2": ["C3", "A2"],
        ...
    },
    "electrode_positions": {
        # standard_1005.elc에서 로드
        "F3": [x, y, z],
        "C3": [x, y, z],
        ...
    }
}
```

### LMDB 저장 구조
```python
# Key: "{subject_id}_{epoch_index}"
key = "subject085_0"

# Value: pickled dictionary
value = {
    "signal": np.array (6, 30, 500),  # float32
    "label": int (0-4),  # W=0, N1=1, N2=2, N3=3, REM=4
    "elc_info": dict,
    "metadata": {
        "subject_id": "subject085",
        "epoch_index": 0,
        "original_sampling_rate": 200,
        "target_sampling_rate": 500,
        "epoch_length_sec": 30
    }
}
```

---

## 참고사항

### 데이터 품질
- ✅ PSG 표준 기록 (병원급 품질)
- ✅ 전문가 라벨링 (수면 전문의)
- ✅ 채널 정렬 완료 (6개 고정)
- ⚠️ 정규화 없음 (모델 학습 시 적용)

### 주의사항
1. **Epoch 길이**: 30초 (다른 데이터셋과 다를 수 있음)
2. **클래스 불균형**: N2가 압도적으로 많음 (50%)
3. **Subject-level split**: Data leakage 방지 필수
4. **REM 라벨 변환**: 원본 5 → DIVER 4

### 데이터셋 크기 추정
```
서브젝트당 평균:
- 수면 시간: ~8시간
- Epoch 수: ~480개 (8시간 / 30초)

전체 데이터셋:
- Train: 84명 × 480 = ~40,000 epochs
- Val: 6명 × 480 = ~3,000 epochs
- Test: 10명 × 480 = ~5,000 epochs
- 총: ~48,000 epochs

용량 (LMDB):
- Epoch당: ~50KB (6×30×500 float32 + metadata)
- 총: ~2.4GB (압축 후)
```

---

## 원본 논문 및 참고자료

### 논문
```
Khalighi, S., Sousa, T., Santos, J. M., & Nunes, U. (2016).
ISRUC-Sleep: A comprehensive public dataset for sleep researchers.
Computer Methods and Programs in Biomedicine, 124, 180-192.
DOI: 10.1016/j.cmpb.2015.10.013
```

### 데이터셋
- **공식 사이트**: https://sleeptight.isr.uc.pt/
- **PhysioNet**: https://physionet.org/content/isruc-sleep/1.0.0/
- **GitHub**: https://github.com/sleeptight-dataset/ISRUC-Sleep

### 관련 프로젝트
- **CBraMod**: https://github.com/your-org/CBraMod (Sleep staging model)

---

## 버전 정보
- **작성일**: 2025-11-21
- **데이터셋 버전**: ISRUC-Sleep Subgroup 1 v1.0
- **전처리 버전**: DIVER 적용 v1.0
- **작성자**: Bohee Lee

---

## 라이센스
- **데이터 사용**: 연구 목적으로 자유롭게 사용 가능
- **인용 필수**: 논문 발표 시 원본 논문 인용 필요
- **Open Access**: Public dataset
