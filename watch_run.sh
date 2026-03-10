#!/bin/bash
# watch_run.sh
# ---------------------------------------------------------
# Yoon Ina's Academic Watch - One-Click Run Script
# ---------------------------------------------------------

echo "========================================================="
echo "🎙️ Yoon Ina's Academic Watch 🏃‍♀️💨"
echo "========================================================="

# 1. Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment 'venv' not found."
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 2. Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# 3. Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "✅ Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# 4. Check for GEMINI_API_KEY environment variable
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY is not set in the environment!"
    echo "Gemini might fail to generate English questions."
    echo "You can set it with: export GEMINI_API_KEY='your_key_here'"
    echo "Proceeding anyway..."
    sleep 2
fi

# 5. Interactive prompt for chunk duration
DEFAULT_DURATION=30
echo -n "⏱️  분석 주기를 몇 초 단위로 설정하시겠습니까? (기본값: $DEFAULT_DURATION): "
read USER_INPUT

if [ -z "$USER_INPUT" ]; then
    CHUNK_DURATION=$DEFAULT_DURATION
else
    if [[ "$USER_INPUT" =~ ^[0-9]+$ ]]; then
        CHUNK_DURATION=$USER_INPUT
    else
        echo "⚠️  잘못된 입력입니다. 기본값인 ${DEFAULT_DURATION}초로 설정합니다."
        CHUNK_DURATION=$DEFAULT_DURATION
    fi
fi

echo -n "🎯 이번 발표/회의에서 특별히 집중 분석할 키워드나 관점이 있나요? (엔터 입력 시 기본 모드): "
read USER_FOCUS

# 7. Run the python script
echo "✅ Starting watch_and_analyze.py (Chunk: ${CHUNK_DURATION}s)..."
if [ -z "$USER_FOCUS" ]; then
    python watch_and_analyze.py --chunk ${CHUNK_DURATION}
else
    python watch_and_analyze.py --chunk ${CHUNK_DURATION} --focus "$USER_FOCUS"
fi

# 8. Deactivate when done
deactivate
echo "✅ Virtual environment deactivated. Goodbye, Professor! 👋"
