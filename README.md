# 👁️ Watch Me Agent (v0.0.1)

**Watch Me Agent**는 발표자의 웹캠 영상(Multi-frame GIF)과 실시간 오디오를 융합하여, **Gemini 3.1 Pro Preview**의 막강한 멀티모달 능력을 통해 발표 내용, 비언어적 맥락, 핵심 질문 사항 등을 리얼타임으로 심층 분석하는 AI 조교 시스템입니다. 

개발자: **김태경 & 윤이나**

---

## 🚀 Antigravity(에이전트)를 통한 완전 자동 환경 구축 가이드

코드나 패키지를 직접 설치하실 필요가 없습니다! 이 저장소의 주소를 복사하여 Antigravity(또는 호환되는 개발 에이전트 인터페이스)에게 전달해 보세요.

**💡 복사 후 에이전트에게 이렇게 명령해 보세요:**
> "이 레포지토리(https://github.com/drtagkim/watch_me_agent)를 가져와서 내 PC에 설치해 주고, 필요한 파이썬 패키지(OpenCV, sounddevice, google-genai, Pillow 등)를 알아서 세팅해 줘."

### 환경 변수 세팅 (필수 API Key)
에이전트가 코드를 실행하기 전, 다음 환경변수가 등록되어 있어야 합니다. (보안상 저장소에는 포함되어 있지 않습니다)

```bash
export GEMINI_API_KEY="당신의_실제_제미나이_API_키"
```

## 🛠 주요 핵심 기능
1. **Continuous 3-Second GIF Capture:** 단순한 1장의 정지 화면이 아닌, 3초 동안의 다중 프레임 움직임을 캡처하여 애니메이션(GIF) 형태로 병합 및 Gemini에게 전송합니다.
2. **Micro/Macro Questioning:** 화자의 STT(오디오 스크립트)와 영상의 맥락을 교차 파악하여 구체적/지엽적 허점을 찌르는 질문(Micro)과, 비즈니스/철학적 관점의 큰 그림 질문(Macro)을 실시간 도출합니다.
3. **Dynamic Context Formatting:** `--focus` 옵션을 통해 심사 분야(예: 투자 피칭, 학술 심사 등)를 설정하면, Gemini 모델 스스로 해당 목적에 최적화된 맞춤형 분석 지침을 도출합니다.
4. **Volume Control:** Numpy 연산을 거친 1차 무음 필터링과 판단 모듈에서의 2차 잡음 필터링으로 낭비성 API 콜을 선제적으로 차단합니다.

## 🏃 실행 방법
해당 툴의 주 목적은 백그라운드 구동이며, 기본 10초 국면(Chunk) 단위로 아래처럼 작동합니다.
```bash
python watch_and_analyze.py --chunk 10 --focus "투자 피칭"
```

## 🤝 버전 로그
- **v0.0.1** (2026-03-10): 최초 멀티모달(Native Audio ↔ Video GIF) 로직 통합 릴리즈. Mlx-whisper 및 Qwen 제거 후 Gemini 3.1 단일 모델 구축 완료.
