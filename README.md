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
