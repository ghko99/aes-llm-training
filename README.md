# AES LLM Training

Kanana LLM 기반 자동 에세이 채점(AES) 모델 LoRA 파인튜닝 코드.

8개 루브릭(과제충실성, 설명명료성, 설명구체성, 설명적절성, 문장연결성, 글통일성, 어휘적절성, 어법적절성)별 1-9점 채점 + 피드백 생성.

## 파일 구조

```
.
├── train.sh                  # 학습 실행 스크립트
├── train.py                  # 메인 학습 코드 (모델 로드, 데이터셋, 학습 루프)
├── trainer.py                # AESTrainer (CE + NTL/WNTL + SAL 커스텀 로스)
├── collator.py               # 데이터 콜레이터 (chat template 포맷팅, 라벨 마스킹)
├── number_token_loss.py      # Number Token Loss 구현
├── number_tokenizer.py       # 숫자 토큰 처리용 토크나이저
├── inference.py              # 추론 코드
├── evaluate.py               # 평가 코드
├── __init__.py
└── aes_datasets/             # 데이터셋 (Git LFS)
    ├── train.jsonl           # 학습 데이터 (64,017개)
    ├── valid.jsonl           # 검증 데이터 (8,000개)
    ├── test.jsonl
    ├── test_14_1.jsonl       # 테스트 (4,000개)
    ├── test_14_2.jsonl       # 테스트 (2,002개)
    └── test_14_3.jsonl       # 테스트 (2,002개)
```

## 학습 환경

- Unsloth + 4-bit QLoRA
- Kanana (LLaMA 아키텍처, 8B)
- W&B 로깅
- EarlyStopping (patience=3)

## 환경 설정

```bash
# 1. 레포 클론
git clone https://github.com/ghko99/aes-llm-training.git
cd aes-llm-training

# 2. conda 환경 생성
conda create -n llm python=3.10 -y
conda activate llm

# 3. PyTorch 설치 (CUDA 버전에 맞게 선택)
# CUDA 12.8 예시:
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# 다른 CUDA 버전은 https://pytorch.org/get-started/locally/ 참고

# 4. 나머지 패키지 설치
pip install -r requirements.txt

# 5. 데이터셋 다운로드 (Git LFS)
git lfs pull

# 6. train.py의 MODEL_PATH를 실제 모델 경로로 수정
# MODEL_PATH = "/path/to/kanana"  (train.py:41)
```

## 실행 방법

```bash
# 1. 상위 디렉토리에서 모듈로 실행
cd ..
./aes-llm-training/train.sh

# 또는 직접 실행
python -m aes-llm-training.train \
    --max_seq_length 2560 \
    --batch_size 1 \
    --grad_accum 32 \
    --lr 2e-4 \
    --epochs 10 \
    --lora_r 16 \
    --lora_alpha 32
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--max_seq_length` | 1536 | 최대 시퀀스 길이 |
| `--batch_size` | 4 | 배치 사이즈 |
| `--grad_accum` | 8 | Gradient accumulation steps |
| `--lr` | 2e-4 | Learning rate |
| `--epochs` | 10 | 학습 에포크 |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--no_ntl` | - | NTL 로스 비활성화 |
| `--use_sal` | - | SAL 로스 활성화 |
| `--no_weighted_ntl` | - | 가중 NTL 비활성화 |
| `--resume` | - | 체크포인트에서 재개 |

## Loss 구성

- **CE**: Cross-Entropy (항상 활성)
- **NTL**: Number Token Loss - 점수 토큰에 대한 MSE
- **WNTL**: Weighted NTL - 클래스 불균형 보정 가중치 적용
- **SAL**: Semantic Alignment Loss - 피드백 토큰 임베딩 거리
- **Dynamic weighting**: `0.5 * (CE + (CE/aux) * aux)`

## 실험 결과 (14-1 데이터셋 기준)

AI Hub 서술형 평가 데이터셋(14-1) 단일 학습 기준, 손실 조합별 성능 비교.

### Overall Metrics

| Loss | QWK | LWK | MAE | RMSE | Exact Acc | Adj Acc (+-1) |
|------|-----|-----|-----|------|-----------|-------------|
| CE | 0.7003 | 0.4949 | 1.017 | 1.418 | 0.3479 | 0.7338 |
| NTL | 0.7334 | 0.5477 | 0.932 | 1.363 | 0.4045 | 0.7533 |
| **WNTL** | **0.7406** | **0.5496** | 0.958 | 1.380 | 0.3891 | 0.7477 |
| SAL | 0.6829 | 0.4743 | 1.044 | 1.440 | 0.3345 | 0.7251 |
| NTL+SAL | 0.7347 | 0.5416 | **0.929** | **1.342** | **0.3991** | **0.7553** |
| WNTL+SAL | 0.7247 | 0.5295 | 0.979 | 1.399 | 0.3805 | 0.7391 |

- **QWK (Quadratic Weighted Kappa)**: WNTL이 0.7406으로 최고
- **MAE / RMSE**: NTL+SAL이 가장 낮은 오차 (MAE 0.929, RMSE 1.342)
- **Adj Acc (+-1)**: NTL+SAL이 75.53%로 최고
- SAL 단독 사용은 CE보다도 성능 하락

### Per-Rubric QWK

| Loss | 과제충실성 | 설명명료성 | 설명구체성 | 설명적절성 | 문장연결성 | 글통일성 | 어휘적절성 | 어법적절성 | Avg |
|------|-----------|-----------|-----------|-----------|-----------|---------|-----------|-----------|-----|
| CE | 0.6005 | 0.5031 | 0.4757 | 0.9589 | 0.2870 | 0.5647 | 0.2994 | 0.7900 | 0.5599 |
| NTL | 0.6550 | 0.5797 | 0.5446 | 0.9620 | 0.3486 | 0.6024 | 0.3470 | 0.8100 | 0.6062 |
| **WNTL** | **0.6585** | **0.5915** | **0.5599** | 0.9603 | **0.4094** | **0.6228** | **0.4090** | **0.8377** | **0.6311** |
| SAL | 0.5675 | 0.4689 | 0.4488 | 0.9553 | 0.2715 | 0.5430 | 0.2918 | 0.7322 | 0.5349 |
| NTL+SAL | 0.6362 | 0.5457 | 0.5303 | 0.9602 | 0.3562 | 0.5938 | 0.3702 | 0.8318 | 0.6031 |
| WNTL+SAL | 0.6413 | 0.5467 | 0.5318 | 0.9599 | 0.3649 | 0.6006 | 0.3862 | 0.8179 | 0.6062 |

- WNTL이 8개 루브릭 중 7개에서 최고 성능 (설명적절성 제외)
- 특히 **문장연결성** (0.2870 -> 0.4094), **어휘적절성** (0.2994 -> 0.4090)에서 CE 대비 큰 폭 개선
- 설명적절성은 모든 모델에서 0.95+ (쉬운 루브릭)

### Per-Rubric MAE

| Loss | 과제충실성 | 설명명료성 | 설명구체성 | 설명적절성 | 문장연결성 | 글통일성 | 어휘적절성 | 어법적절성 | Avg |
|------|-----------|-----------|-----------|-----------|-----------|---------|-----------|-----------|-----|
| CE | 1.071 | 1.254 | 1.199 | 0.313 | 1.128 | 1.175 | 1.084 | 0.915 | 1.017 |
| NTL | 1.025 | 1.102 | 1.067 | 0.247 | 1.072 | 1.098 | 1.110 | 0.738 | 0.932 |
| WNTL | 1.054 | 1.143 | 1.150 | 0.267 | 1.105 | 1.120 | 1.098 | **0.727** | 0.958 |
| SAL | 1.092 | 1.252 | 1.227 | 0.337 | 1.143 | 1.187 | 1.091 | 1.026 | 1.044 |
| NTL+SAL | **1.032** | **1.121** | **1.104** | **0.261** | **1.062** | **1.096** | **1.025** | 0.729 | **0.929** |
| WNTL+SAL | 1.059 | 1.196 | 1.151 | 0.277 | 1.163 | 1.117 | 1.097 | 0.773 | 0.979 |

### 요약

- **QWK 기준 최고 성능**: WNTL (CE + Weighted Number Token Loss)
- **오차 기준 최고 성능**: NTL+SAL (CE + Number Token Loss + Semantic Alignment Loss)
- NTL 계열 손실 추가 시 CE 대비 전반적으로 성능 향상
- SAL 단독 사용은 오히려 성능 저하, NTL과 조합 시 오차 감소에 기여
- WNTL의 클래스 가중치가 소수 클래스(낮은/높은 점수) 예측 정확도를 높여 QWK 향상에 효과적

## 데이터셋 형식

```json
{
  "system": "에세이 채점기. 8개 루브릭(...)별 1-9점 채점 후 피드백을 작성한다.",
  "user": "질문: ...\n에세이: ...\n핵심 키워드: ...",
  "assistant": "4 4 3 3 4 4 4 4\n\n### Feedback:\n- 과제 수행의 충실성:\n ..."
}
```

## 참고

- `MODEL_PATH`를 실제 Kanana 모델 경로로 수정 필요 (`train.py:41`)
- 데이터셋은 Git LFS로 관리 — clone 후 `git lfs pull` 필요
- PyTorch는 서버의 CUDA 버전에 맞게 별도 설치 권장
