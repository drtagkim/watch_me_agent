#!/bin/bash
# MacOS M3 16GB 최적화 Qwen3-VL-8B 실행 스크립트

echo "🚀 Qwen3-VL-8B 서버를 시작합니다... (M3 GPU 최적화)"

# 사용자의 기존 Qwen 디렉토리 내 llama.cpp 경로 및 모델을 가정합니다.
# 만약 경로가 다르면 교수님이 쉽게 수정할 수 있도록 변수로 분리!
LLAMA_DIR="/Users/tagg/dev/Qwen/llama.cpp"
MODEL_PATH="/Users/tagg/dev/Qwen/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH="/Users/tagg/dev/Qwen/mmproj-Qwen3VL-8B-Instruct-F16.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 에러: 모델을 찾을 수 없습니다 ($MODEL_PATH)"
    exit 1
fi

# M3 (16GB) 환경 맞춤형 파라미터 (명령줄 스트리밍 속도 최적화)
# -c 4096: 컨텍스트 윈도우 한계 설정 (16GB 메모리)
# -cb: continuous batching (연속 스트리밍 지원)

cd "$LLAMA_DIR" || exit

./llama-server \
  -m "$MODEL_PATH" \
  --mmproj "$MMPROJ_PATH" \
  --port 8080 \
  -c 4096 \
  -cb \
  -ngl 99 \
  --host 127.0.0.1

echo "✅ 서버 구동 완료. http://localhost:8080/v1 엔드포인트 대기 중"
