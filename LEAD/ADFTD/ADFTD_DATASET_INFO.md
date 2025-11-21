# ADFTD (Alzheimer's Disease & Frontotemporal Dementia) Dataset - 전처리 정보 총정리

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
- **데이터셋명**: ADFTD (Alzheimer's Disease & Frontotemporal Dementia EEG Dataset)
- **출처**: LEAD (Learning EEG Analysis for neurodegenerative Diseases) Project
- **목적**: 알츠하이머 및 전두측두엽 치매 진단
- **대상**: 치매 환자 및 정상 대조군
- **데이터 타입**: 두피 뇌파 (Scalp EEG)

### 서브젝트 정보
- **전체 서브젝트 수**: 88명
- **클래스 분포**:
  - **CN (Cognitively Normal)**: 정상 대조군
  - **AD (Alzheimer's Disease)**: 알츠하이머병
  - **FTD (Frontotemporal Dementia)**: 전두측두엽 치매
- **Multi-class Classification**: 3-way classification (CN vs AD vs FTD)

---

## 원본 데이터 구조

### 파일 구조
```
/data/ADFTD/
├── CN/
│   ├── subject001_CN.edf
│   ├── subject002_CN.edf
│   └── ...
├── AD/
│   ├── subject001_AD.edf
│   ├── subject002_AD.edf
│   └── ...
└── FTD/
    ├── subject001_FTD.edf
    ├── subject002_FTD.edf
    └── ...
```

### 채널 정보

#### 전극 시스템
- **시스템**: 10-20 International System
- **채널 수**: 19개 (표준 10-20 채널)
- **Reference**: Average reference

#### 19개 채널 목록
```python
channels = [
    # Frontal
    "FP1", "FP2", "F3", "F4", "F7", "F8", "FZ",

    # Central
    "C3", "C4", "CZ",

    # Temporal
    "T3", "T4", "T5", "T6",

    # Parietal
    "P3", "P4", "PZ",

    # Occipital
    "O1", "O2"
]
```

### Recording 정보
- **Sampling Rate**: 256 Hz
- **파일 형식**: EDF (European Data Format)
- **Recording 길이**: 환자마다 다름 (일반적으로 5-30분)
- **데이터 타입**: Float (μV 단위)
- **상태**: Eyes-closed resting state

---

## 전처리 파이프라인

### 파이프라인 구조
```
Raw EDF Files → Channel Extraction → Clipping → Segmentation → LMDB
```

### Stage 1: 원시 데이터 로드 및 채널 추출

#### 목적
- EDF 파일에서 19개 표준 채널 추출
- 클래스 라벨 파싱 (CN/AD/FTD)
- 품질 검증

#### 처리 과정
1. **EDF 로드** (`load_raw_edf_singlechannel()`)
   - MNE 라이브러리 사용
   - 19개 채널만 선택
   - Sampling rate 확인 (256 Hz)

2. **채널 정렬**
   - 표준 10-20 순서로 정렬
   - 누락된 채널 처리 (zero-padding 또는 skip)

3. **클래스 라벨 추출**
   - 파일명에서 추출 (`subject001_CN.edf` → label='CN')
   - One-hot encoding: CN=0, AD=1, FTD=2

### Stage 2: Clipping (Artifact Removal)

#### 목적
- Artifact(잡음) 제거
- 품질 낮은 구간 제외
- 유효한 신호만 추출

#### Clipping 기준
```python
# clip_extraction_utils.py
def detect_artifacts(signal):
    # 1. Amplitude clipping
    threshold_high = 100  # μV
    threshold_low = -100  # μV

    # 2. Gradient clipping (급격한 변화 감지)
    gradient = np.diff(signal)
    threshold_gradient = 50

    # 3. Flatline detection (신호 정체)
    std_window = np.std(signal[window])
    threshold_std = 5

    return clean_segments
```

#### 출력
- Artifact가 제거된 깨끗한 신호 구간만 추출
- 연속적인 깨끗한 구간들

### Stage 3: 세그먼테이션

#### 목적
- 연속 신호를 고정 길이 세그먼트로 분할
- 모델 입력 형태로 변환

#### 세그먼트 파라미터
```python
SEGMENT_LENGTH = 10  # 초
SAMPLING_RATE = 256  # Hz
SEGMENT_SAMPLES = 2560  # 10초 × 256Hz
SLIDE_STEP = 2560  # Non-overlapping (10초 step)
```

#### 세그먼트 생성
```python
for i in range(0, len(signal), SEGMENT_SAMPLES):
    segment = signal[:, i:i+SEGMENT_SAMPLES]  # (19, 2560)

    # Quality check
    if is_good_quality(segment):
        segments.append(segment)
```

### Stage 4: Train/Val/Test Split

#### 분할 전략
- **Subject-level split**: 환자 단위로 분할 (data leakage 방지)
- **비율**: Train 70% / Val 15% / Test 15%
- **Stratified**: 각 클래스 비율 유지

```python
split_ratio = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15
}

# 클래스별로 stratified split
for class_label in ['CN', 'AD', 'FTD']:
    subjects = get_subjects(class_label)
    train, val, test = stratified_split(subjects, split_ratio)
```

---

## 라벨링 및 클래스 불균형

### 라벨 정의
- **Label 0 (CN)**: Cognitively Normal (정상 대조군)
- **Label 1 (AD)**: Alzheimer's Disease (알츠하이머병)
- **Label 2 (FTD)**: Frontotemporal Dementia (전두측두엽 치매)

### 클래스 불균형

#### 원본 데이터 분포
```
CN:  30명 (~34%)
AD:  35명 (~40%)
FTD: 23명 (~26%)

총: 88명
```

#### 불균형 해결 방법
1. **Class weights**: 학습 시 loss에 class weight 적용
2. **Balanced sampling**: 각 클래스에서 균등하게 샘플링
3. **Stratified split**: Train/Val/Test에 클래스 비율 유지

---

## 원본 vs DIVER 적용 비교표

| 구분 | 항목 | 원본 (ADFTD Original) | 수정 (DIVER 적용) |
|------|------|----------------------|-------------------|
| **데이터셋** | 전체 서브젝트 수 | 88명 | **88명** (동일) |
| | 클래스 | CN, AD, FTD (3-class) | **3-class** (동일) |
| | Train/Val/Test | 70% / 15% / 15% | **70% / 15% / 15%** (동일) |
| **원본 데이터** | Sampling Rate | 256 Hz | **256 Hz** (동일) |
| | 파일 형식 | EDF | **EDF** (동일) |
| | 채널 수 | 19개 | **19개** (동일) |
| **채널 시스템** | 전극 배치 시스템 | 10-20 System | **10-20 System** (동일) |
| | Reference | Average reference | **Average reference** (동일) |
| | 채널 목록 | FP1, FP2, F3, F4, ... | **동일** |
| | ELC 파일 사용 | ❌ 없음 | ✅ **사용** (standard_1005.elc) |
| **전처리** | Clipping (Artifact 제거) | ✅ 적용 | ✅ **적용** (동일) |
| | 세그먼트 길이 | 10초 | **10초** (동일) |
| | 슬라이딩 윈도우 | Non-overlapping (10초 step) | **Non-overlapping** (동일) |
| **리샘플링** | 타겟 Sampling Rate | - (256 Hz 유지) | ✅ **500 Hz** |
| | 리샘플링 방법 | - | **scipy.signal.resample** |
| | Reshape | - | ✅ **(19, 5000) → (19, 10, 500)** |
| **라벨링** | 라벨 종류 | 3-class (CN, AD, FTD) | **3-class** (동일) |
| | Label 0 | CN (정상) | **CN** (동일) |
| | Label 1 | AD (알츠하이머) | **AD** (동일) |
| | Label 2 | FTD (전두측두엽 치매) | **FTD** (동일) |
| **최종 출력** | Shape | **(19, 2560)** | ✅ **(19, 10, 500)** |
| | | 19채널, 2560샘플 | 19채널, 10×1초, 500샘플/초 |
| | 데이터 구조 | `{"signal": array, "label": int}` | `{"signal": array, "label": int, "elc_info": dict}` |
| | 저장 형식 | `.pkl` or `.h5` | ✅ **LMDB** |
| **정규화** | Z-score 정규화 | ❌ 없음 | ❌ **없음** (모델에서 처리) |

---

## 구현 상세

### Shape 변환 과정
```python
# 원본 (256Hz)
signal_256 = np.array (19, 2560)  # 19채널 × 10초 × 256Hz

# Step 1: 리샘플링 (256Hz → 500Hz)
from scipy.signal import resample
signal_500 = resample(signal_256, 5000, axis=1)  # (19, 5000)

# Step 2: Reshape (10개 1초 세그먼트)
signal_final = signal_500.reshape(19, 10, 500)  # (19, 10, 500)
```

### ELC 파일 구조
```python
elc_info = {
    "channel_names": [
        "FP1", "FP2", "F3", "F4", "F7", "F8", "FZ",
        "C3", "C4", "CZ",
        "T3", "T4", "T5", "T6",
        "P3", "P4", "PZ",
        "O1", "O2"
    ],
    "electrode_positions": {
        # standard_1005.elc에서 로드
        "FP1": [x, y, z],
        "FP2": [x, y, z],
        ...
    }
}
```

### LMDB 저장 구조
```python
# Key: "{subject_id}_{segment_index}"
key = "subject001_CN_0"

# Value: pickled dictionary
value = {
    "signal": np.array (19, 10, 500),  # float32
    "label": int (0, 1, or 2),  # CN=0, AD=1, FTD=2
    "elc_info": dict,
    "metadata": {
        "subject_id": "subject001_CN",
        "segment_index": 0,
        "original_sampling_rate": 256,
        "target_sampling_rate": 500,
        "diagnosis": "CN"  # or "AD", "FTD"
    }
}
```

---

## 참고사항

### 데이터 품질
- ✅ EDF 파일 무결성 확인됨
- ✅ Artifact 제거 (Clipping) 적용
- ✅ 채널 정렬 완료 (19개 고정)
- ⚠️ 정규화 없음 (모델 학습 시 적용)

### 주의사항
1. **환자 정보 보호**: Subject ID만 사용 (개인정보 제거됨)
2. **클래스 불균형**: Stratified split으로 완화
3. **Subject-level split**: Data leakage 방지
4. **ELC 매핑**: 19개 단극 전극 위치 저장

### 데이터셋 크기 추정
```
서브젝트당 평균:
- Recording 길이: ~15분
- 세그먼트 수 (clipping 후): ~50-80개

전체 데이터셋:
- Train: ~62명 × 65 = ~4,000 샘플
- Val: ~13명 × 65 = ~850 샘플
- Test: ~13명 × 65 = ~850 샘플
- 총: ~5,700 샘플

용량 (LMDB):
- 샘플당: ~50KB (19×10×500 float32 + metadata)
- 총: ~300MB (압축 후)
```

---

## 원본 논문 및 참고자료

### 논문
```
Park, J. E., et al. (2024).
LEAD: Learning EEG Analysis for neurodegenerative Diseases
Journal of Neural Engineering (예정)
```

### 데이터셋
- **GitHub**: https://github.com/your-org/ADFTD-dataset (비공개)
- **PhysioNet**: (예정)

### 관련 프로젝트
- **LEAD Project**: https://lead-project.org

---

## 버전 정보
- **작성일**: 2025-11-21
- **데이터셋 버전**: ADFTD v1.0
- **전처리 버전**: DIVER 적용 v1.0
- **작성자**: Bohee Lee

---

## 라이센스
- **데이터 사용**: 연구 목적으로만 사용 가능
- **재배포 금지**: 원본 데이터 재배포 불가
- **인용 필수**: 논문 발표 시 반드시 인용
