import cv2
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import base64
from google import genai
import argparse
import sys
import threading
import datetime
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# antigravity_ui 경로 추가
antigravity_ui_path = os.path.abspath("/Users/tagg/Library/CloudStorage/GoogleDrive-masan.korea@gmail.com/내 드라이브/Development/2026-02-10(Antigravity-coding-knowledge)")
if antigravity_ui_path not in sys.path:
    sys.path.append(antigravity_ui_path)

from antigravity_ui import (
    Spinner, print_success, print_error, print_warning, print_info,
    Color, style, print_styled
)

# 전역 상태 변수
is_running = True
cumulative_transcripts = []
cumulative_reports = []
GLOBAL_DYNAMIC_PROMPT = ""
TEMP_DIR = "temp"

# 하드웨어 스트림 자원
cap = None
audio_stream = None

# 실시간 수집 버퍼
audio_buffer = []
audio_buffer_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()

def get_macbook_mic_index():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev['name']
        if ('MacBook' in name or 'Built-in' in name) and dev['max_input_channels'] > 0:
            return i
    print_warning("'MacBook' mic not found. Using default input.")
    return sd.default.device[0]

def cleanup_hardware():
    """모든 하드웨어 리소스를 즉시 반환합니다."""
    global audio_stream, cap
    if audio_stream is not None:
        try:
            audio_stream.stop()
            time.sleep(0.1) # PortAudio 콜백이 자연스럽게 종료될 시간 벌기
            audio_stream.close()
        except Exception:
            pass
        finally:
            audio_stream = None

    if cap is not None and cap.isOpened():
        try:
            cap.release()
            print_success("카메라(비디오) 자원을 안전하게 반환했습니다.")
        except Exception:
            pass
        finally:
            cap = None

def cleanup_temp_dir():
    """temp 폴더 내의 모든 파일을 삭제합니다 (폴더 자체는 유지)."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print_error(f"임시 파일 삭제 실패 ({file_path}): {e}")
    print_success(f"임시 폴더({TEMP_DIR}/) 정리를 완료했습니다.")

def audio_callback(indata, frames, time_info, status):
    """지속적으로 불리는 오디오 콜백"""
    if is_running:
        with audio_buffer_lock:
            audio_buffer.append(indata.copy())

def video_capture_loop():
    """웹캠을 상시 유지하며 자동 노출(밝기)을 맞추고 최신 프레임을 갱신하는 루프"""
    global cap, latest_frame, is_running
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print_error("웹캠을 초기화할 수 없습니다.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 워밍업: 카메라 켜진 직후의 어두운 프레임들을 소진시켜 자동 노출 맞출 시간 벌기
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    while is_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        # 약 30fps 수준으로 지속적으로 버퍼를 비워주어야 자동 노출과 실시간성이 유지됨
        time.sleep(0.03)

def handle_exit(signum=None, frame=None):
    """Ctrl+C 또는 강제 종료 시그널 처리"""
    global is_running
    is_running = False
    print()
    print_warning("(종료 신호 감지: 현재 진행 중인 분석까지만 마치고 종료합니다)")

def process_chunk(recording, frames, exact_timestamp, chunk_idx, fs=16000, log_file="", image_dir=""):
    audio_file = os.path.join(TEMP_DIR, f"temp_presentation_chunk_{chunk_idx}.wav")
    wav.write(audio_file, fs, recording)
    
    # Raw Image 프레임들을 GIF로 저장
    frame_path = ""
    if image_dir and frames:
        frame_filename = f"frame_{chunk_idx}_{exact_timestamp.replace(':', '-')}.gif".replace(" ", "_")
        frame_path = os.path.join(image_dir, frame_filename)
        
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        # duration=3000ms (3초마다 전환)
        pil_images[0].save(frame_path, save_all=True, append_images=pil_images[1:], duration=3000, loop=0)
    
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as af:
            audio_bytes = af.read()
    else:
        return
        
    # Python 기반 무음 필터링 (완전 정적인 앰비언트 노이즈 이하는 버림)
    max_amp = np.max(np.abs(recording))
    if max_amp < 0.005:  # 마이크에 소리가 안 들어오는 수준의 float32 threshold
        print()
        print_info(f"[{exact_timestamp}] [분석 스레드] 국면 {chunk_idx}: 유의미한 소리 미감지 (max_amp={max_amp:.4f}). 생략.")
        if os.path.exists(audio_file): os.remove(audio_file)
        return
        
    print()
    print_info(style(f"[{exact_timestamp}] [분석 스레드] 국면 {chunk_idx}: Gemini 3.1 Pro Preview로 오디오와 영상 원본 동시 전송 중...", Color.CYAN))
    
    try:
        # 영상 해상도 축소 및 API 연동 (Gemini        # API Key 가져오기 (환경변수 의존)
        api_key = os.environ.get("GEMINI_API_KEY")    
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
            
        gemini_client = genai.Client(api_key=api_key)
        
        unified_prompt = f"""
        당신은 수석 연구조교 '윤이나'입니다.
        현재 시각은 {exact_timestamp} 입니다. 이것은 전체 발표 중 방금 흘러간 최신 분량의 국면 {chunk_idx} 상황입니다.
        
        당신에게는 방금 전 발표의 '3초 간격 화면 캡처 이미지들'과 '현장 오디오 파일'이 리얼타임으로 동시에 제공되었습니다.
        단순 툴이 아니라 윤이나의 자아로서, 영상 상황(시각 자료, 슬라이드 등)과 오디오 내용(화자의 이야기)을 직관적으로 '통섭 파악'하세요.
        
        아래의 두 가지 파트로 나누어 출력 파일 구조를 반드시 지키십시오.

        <TRANSCRIPTION>
        (오디오 파일에서 들린 화자의 발언을 스크립트 그대로 전사하세요. 외국어라면 그대로 두고, 무의미한 잡음만 있다면 [무음] 이라고 적으세요.)
        </TRANSCRIPTION>

        <REPORT>
        [맞춤형 핵심 분석 지침]
        {GLOBAL_DYNAMIC_PROMPT}

        위의 맞춤형 지침을 우선 처리하되, 추가로 아래 임무도 이 REPORT 안에 녹여 넣으세요:
        1. **번역/언어 체크**: 화자가 한국어가 아닌 다른 언어를 썼다면, 맥락에 맞게 유려한 한국어로 번역/요약해 주세요.
        2. **핵심 질문 2개 초안**: 화면 자료(+텍스트)의 시각적 흐름과 오디오 맥락의 허점을 교차 파악하여 예리하게,
           - Micro Question (구체적/지엽적 포인트를 찌르는 질문 1개, 국/영문 병기)
           - Macro Question (전대미문의 큰 그림 또는 철학/BM적 질문 1개, 국/영문 병기)

        마크다운을 사용해 학술적이고 명료하게, 그리고 자신감 넘치는 윤이나의 톤앤매너로 답변하세요.
        </REPORT>
        """
        
        # 여러 장의 이미지를 순차적으로 Part로 변환
        api_parts = [unified_prompt]
        for f in frames:
            resized_frame = cv2.resize(f, (472, 354)) # 해상도 최적화 (가로 472, 세로 354로 4:3 비율 유지)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            _, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
            api_parts.append(genai.types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
            
        api_parts.append(genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
        
        response = gemini_client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=api_parts
        )
        unified_result = response.text
        
        if os.path.exists(audio_file):  # 분석 끝났으니 임시 오디오 삭제
            os.remove(audio_file)
            
        transcription = "[음성 전사 내용 누락]"
        report_text = unified_result
        
        if "<TRANSCRIPTION>" in unified_result and "</TRANSCRIPTION>" in unified_result:
            transcription = unified_result.split("<TRANSCRIPTION>")[1].split("</TRANSCRIPTION>")[0].strip()
            
        if "<REPORT>" in unified_result and "</REPORT>" in unified_result:
            report_text = unified_result.split("<REPORT>")[1].split("</REPORT>")[0].strip()

        # 전사 내용 중에 진짜 [무음]만 있다면 스킵 처리
        if transcription in ["[무음]", "무음"]:
             print()
             print_warning(f"[{exact_timestamp}] [분석 스레드] 국면 {chunk_idx}: 오디오가 잡음/무음 판별되어 통섭 무시됨.")
             return
             
        cumulative_transcripts.append(f"[{exact_timestamp}] {transcription}")
        cumulative_reports.append(f"### 국면 {chunk_idx} 분석 ({exact_timestamp})\n{report_text}\n")
        
        # 3. 실시간 터미널 출력 및 마크다운 Append
        print()
        print_styled(f"==== 💡 [Gemini 통섭 분석 완료] 국면 {chunk_idx} ====", Color.GREEN, Color.BOLD)
        print_styled(f"[STT 전사]: {transcription[:80]}...", Color.MAGENTA)
        print_styled("-" * 40, Color.DIM)
        print(report_text)
        print_styled("=========================================", Color.DIM)
        print()
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## 국면 {chunk_idx} ({exact_timestamp})\n")
            if frame_path:
                f.write(f"**📸 3초 단위 시각 자료 (GIF Animation):**\n![Frames]({frame_path})\n\n")
            f.write(f"**🗣️ 음성 인식 (Native Audio To Text):**\n> {transcription}\n\n")
            f.write(f"**🧠 AI 즉각 통섭 분석 (Gemini 3.1 Pro Preview):**\n{report_text}\n")
            f.write("---\n")
            
        # 4. 별도의 질문(_question.md) 파일 저장
        question_file = log_file.replace(".md", "_question.md")
        with open(question_file, "a", encoding="utf-8") as qf:
            qf.write(f"\n## 국면 {chunk_idx} ({exact_timestamp})\n")
            qf.write(f"**🗣️ 음성 발화 (맥락 파악용):**\n> {transcription[:100]}...\n\n")
            qf.write(f"{report_text}\n")
            qf.write("---\n")
            
    except Exception as e:
         print_error(f"분석 처리 중 오류 발생: {e}")

def generate_global_summary(session_id):
    if not cumulative_transcripts and not cumulative_reports:
        print()
        print_info("수집된 데이터가 없어 총괄 요약을 생략합니다.")
        return
        
    print()
    print_styled("*"*60, Color.YELLOW, Color.BOLD)
    print_styled("✨ [윤이나의 발표 총괄 최종 종합 분석 중...]", Color.YELLOW, Color.BOLD)
    print_styled("*"*60, Color.YELLOW, Color.BOLD)
    print()
    
    all_transcripts = "\n".join(cumulative_transcripts)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY 환경변수가 설정되지 않아 요약을 생성할 수 없습니다.")
        return
        
    gemini_client = genai.Client(api_key=api_key)
    
    prompt = f"""
    당신은 김태경 교수님의 에이스 수석 연구조교 '윤이나'입니다.
    방금 진행된 긴 학술 발표가 완전히 종료되었습니다. 
    아래는 전체 발표 시간 동안 누적된 시간대별 음성 스크립트 모음입니다.
    
    [전체 발표 스크립트 누적본]:
    {all_transcripts}

    이 내용들을 모두 통합하여 교수님께 '최종 마감 보고서'를 제출해 주세요.
    먼저, 내용 전체를 관통하는 가장 적절하고 센스있는 핵심 제목(Title)을 지어주세요.
    반드시 다음 태그 안에 제목을 작성해 주세요: <TITLE>원하는 제목</TITLE>
    
    그리고 아래 내용을 차례로 작성하세요:
    1. 📌 <strong>발표 총괄 요약</strong>: (3~5줄 핵심)
    2. 🎯 <strong>주요 논점 및 쟁점</strong>:
    3. ⚖️ <strong>최종 종합 평가</strong>: (강점 및 한계점)
    4. 🚀 <strong>[💡 윤이나의 추가 제안]</strong>: 교수님이 앞으로 챙기시면 좋을 후속 작업(Next Step) 2개
    
    자신감 넘치는 조교의 프로페셔널한 어조로 마크다운 형식으로 작성해 주세요. 주의: 제목이나 굵은 글씨 부분에 별표(**) 대신 HTML 태그인 <strong>태그를 사용하세요 (예: <strong>제목</strong>).
    """
    
    try:
        response = gemini_client.models.generate_content_stream(
            model="gemini-3.1-pro-preview",
            contents=prompt,
        )
        
        final_summary = ""
        for chunk in response:
            if chunk.text:
                final_summary += chunk.text
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
                
        print()
        print_styled("*"*60, Color.YELLOW, Color.BOLD)
        
        dynamic_title = "🎓 윤이나의 종합 심층 분석 리포트"
        if "<TITLE>" in final_summary and "</TITLE>" in final_summary:
            dynamic_title = final_summary.split("<TITLE>")[1].split("</TITLE>")[0].strip()
            final_summary = final_summary.replace(f"<TITLE>{dynamic_title}</TITLE>", "").strip()
            final_summary = final_summary.replace("<TITLE>", "").replace("</TITLE>", "").strip()
            
        final_filename = f"Final_Summary_{session_id}.md"
        with open(final_filename, "w", encoding="utf-8") as f:
            f.write(f"# {dynamic_title}\n")
            f.write(f"> **Session ID:** `{session_id}`\n\n")
            f.write(final_summary)
            
        # 실시간 로그와 질문 로그 파일의 제목도 새로 지어진 멋진 제목으로 업데이트
        realtime_log_file = f"Realtime_Log_{session_id}.md"
        question_log_file = realtime_log_file.replace(".md", "_question.md")
        
        def update_log_title(filepath, new_title):
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as file_read:
                    content = file_read.read()
                lines = content.split('\n')
                if len(lines) > 0 and lines[0].startswith("# "):
                    lines[0] = f"# {new_title}"
                with open(filepath, "w", encoding="utf-8") as file_write:
                    file_write.write('\n'.join(lines))
                    
        update_log_title(realtime_log_file, dynamic_title)
        update_log_title(question_log_file, f"🤔 [Q&A] {dynamic_title}")
            
        print()
        print_success(f"최종 총괄 리포트가 안전하게 저장되었습니다: {os.path.abspath(final_filename)}")
        print_success(f"이전 실시간 로그 파일들의 제목도 분석 내용에 맞게 최종 갱신되었습니다.")
        
    except Exception as e:
        print_error(f"총괄 요약 생성 중 오류 발생: {e}")

def generate_dynamic_prompt(focus_area):
    global GLOBAL_DYNAMIC_PROMPT
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    base_default_prompt = "1. 📝 기록 (Record): 핵심 논의 팩트 체크 및 수치/방법론 식별\n2. 📌 분석 (Analysis): 논리 전개 흐름 및 문맥적 가치 심층 분석\n3. ⚖️ 평가 (Evaluation): 주장의 타당성 진단 및 날카로운 보완점 도출"
    
    if not gemini_key:
        print_warning("GEMINI_API_KEY가 없어 기본 분석 지침을 사용합니다.")
        GLOBAL_DYNAMIC_PROMPT = base_default_prompt
        return
        
    focus_text = focus_area if focus_area else "일반 학술 논문 및 연구 발표 심사"
    print()
    print_info(style(f"🧠 [Gemini 3.1 Pro Preview] 사용자의 집중 분석 목표('{focus_text}')를 반영하여 고도화된 맞춤형 지침을 동적 생성 중...", Color.CYAN))
    try:
        gemini_client = genai.Client(api_key=gemini_key)
        
        system_instructions = f"""
당신은 최고 수준의 시스템 프롬프트 전문가입니다.
현재 동료 AI 모델(Gemini)이 웹캠 화상 프레임과 실시간 음성(STT)을 종합해 사용자의 특수 목적에 맞도록 보고서를 써야 합니다.

사용자의 현재 특수 목적/집중 분야는 다음과 같습니다: "{focus_text}"

이 AI가 실시간으로 영상과 음성을 분석할 때 지켜야 할 [맞춤형 핵심 분석 지침] 3가지를 마크다운 리스트 형태로 작성해주세요.
반드시 아래의 구조를 따르되, 사용자의 '특수 목적'에 완벽히 특화된 전문 용어와 판별 기준으로 지시문을 극도로 날카롭게 커스텀하세요.

1. 📝 기록 (Record): (목적에 맞는 핵심 데이터/관찰점/수치/팩트 캡처 지시)
2. 📌 분석 (Analysis): (해당 목적에서 가장 중요한 문맥적 가치, 논리성, 비즈니스/학술적 의미 분석 지시)
3. ⚖️ 평가 (Evaluation): (결정적인 강점, 치명적인 허점, 또는 실질적이고 예리한 피드백 도출 지시)

참고 예시)
- IR/투자 피칭: BM 수익성 및 거시 지표 기록 -> 투자 매력도 및 현실성 분석 -> 경쟁우위 방어력 및 투자 리스크 평가
- 학술 심사: 연구 방법론, 주요 변수 기록 -> 연구의 논리적 독창성 및 가설 검증 분석 -> 한계점 및 학술적/실무적 기여도 진단

불필요한 서론/결론이나 인사말 없이 오직 **1, 2, 3번 지시문 본문만 깔끔한 마크다운 형태**로 출력하세요.
        """
        response = gemini_client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=system_instructions
        )
        GLOBAL_DYNAMIC_PROMPT = response.text.strip()
        print_success(f"동적 분석 지침 생성 완료:\n{style(GLOBAL_DYNAMIC_PROMPT, Color.DIM)}")
    except Exception as e:
        print_warning(f"Gemini 동적 프롬프트 생성 실패 ({e}). 기본 지침 사용.")
        GLOBAL_DYNAMIC_PROMPT = base_default_prompt

def CLI_wait_for_exit():
    global is_running
    # input()은 block이 되므로, Ctrl+C를 선호하지만 둘 다 지원.
    try:
        input()  
        if is_running: # Enter를 쳐서 종료할경우
            print()
            print_warning("(Enter 키 입력 감지: 기록을 중단합니다)")
            handle_exit()
    except EOFError:
        pass

def main_loop(chunk_duration=30, focus_area=""):
    """상시 관찰 메인 루프"""
    global is_running

    # 변수 초기화 (Scope 에러 방지용)
    session_id = None
    realtime_log_file = None
    question_log_file = None
    image_dir = None
    
    # 시스템 시그널 등록 (Ctrl+C 등 강제종료 대비)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # 0. Gemini를 활용한 동적 프롬프트 사전 생성
    generate_dynamic_prompt(focus_area)
    
    # 0.5 임시 폴더 초기화
    cleanup_temp_dir()

    fs = 16000
    mic_idx = get_macbook_mic_index()
    device_name = sd.query_devices(mic_idx)['name']
    
    session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    realtime_log_file = f"Realtime_Log_{session_id}.md"
    question_log_file = realtime_log_file.replace(".md", "_question.md")
    image_dir = f"Raw_Images_{session_id}"
    os.makedirs(image_dir, exist_ok=True)
    
    with open(realtime_log_file, "w", encoding="utf-8") as f:
        f.write(f"# ⏳ [실시간 데이터 분석 진행 중...]\n")
        f.write(f"> **Session ID:** `{session_id}`\n")
        f.write(f"> **시작 시각:** {session_id.replace('_', ' ')}\n")
        f.write("---\n")
        
    with open(question_log_file, "w", encoding="utf-8") as qf:
        qf.write(f"# ⏳ [실시간 핵심 질문 도출 진행 중...]\n")
        qf.write(f"> **Session ID:** `{session_id}`\n")
        qf.write(f"> **시작 시각:** {session_id.replace('_', ' ')}\n")
        qf.write("---\n")
    
    print()
    print_styled("="*70, Color.BLUE, Color.BOLD)
    print_styled("🎙️ [Yoon Ina's Academic Watch] - 연속 스트리밍 모드 CLI", Color.BLUE, Color.BOLD)
    print_styled("="*70, Color.BLUE, Color.BOLD)
    print(style("✔️ 입력 장치: ", Color.GREEN) + f"{device_name} (Index: {mic_idx})")
    print(style("✔️ 분석 간격: ", Color.GREEN) + f"{chunk_duration}초 (1개 국면당)")
    print(style("✔️ 실시간 로그 파일: ", Color.GREEN) + f"{os.path.abspath(realtime_log_file)}")
    print(style("🚨 단축키 안내: ", Color.RED, Color.BOLD) + "실행 중 " + style("[Enter]", Color.WHITE, Color.BOLD) + " 키 또는 " + style("[Ctrl+C]", Color.WHITE, Color.BOLD) + "를 누르면 즉각 기록을 멈추고 안전하게 요약본을 생성한 뒤 하드웨어를 회수합니다.")
    print()
    print_styled("▶️ 하드웨어 연속 캡처 및 분석을 시작합니다...", Color.CYAN, Color.BOLD)
    print()
    
    # 1. Video 백그라운드 캡처 시작
    v_thread = threading.Thread(target=video_capture_loop, daemon=True)
    v_thread.start()
    
    # 2. Audio 백그라운드 콜백 캡처 시작
    global audio_stream
    audio_stream = sd.InputStream(samplerate=fs, channels=1, dtype='float32', device=mic_idx, callback=audio_callback)
    audio_stream.start()
    
    # 3. 종료 대기 스레드
    exit_thread = threading.Thread(target=CLI_wait_for_exit, daemon=True)
    exit_thread.start()
    
    chunk_idx = 1
    cumulative_audio_data = []
    # STT와 LLM 병렬 안전처리 위해 2개~4개 할당 (M3 스레드 효율)
    executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="WatchWorker")
    try:
        while is_running:
            start_time = time.time()
            chunk_frames = []
            
            # 첫 번째 프레임 즉각 캡처
            with frame_lock:
                if latest_frame is not None:
                    chunk_frames.append(latest_frame.copy())
            last_frame_capture_time = time.time()
            
            # chunk_duration 만큼 대기하며 3초 단위 백그라운드 캡처 허용
            spin_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            spin_idx = 0
            while is_running and (time.time() - start_time) < chunk_duration:
                current_time = time.time()
                # 3초마다 프레임 캡처 이벤트
                if (current_time - last_frame_capture_time) >= 3.0:
                    with frame_lock:
                        if latest_frame is not None:
                            chunk_frames.append(latest_frame.copy())
                    last_frame_capture_time = current_time
                    sys.stdout.write(f"\r{style('📸 [찰칵! 화면 포착]', Color.YELLOW, Color.BOLD)} ({len(chunk_frames)} frames)".ljust(50))
                    sys.stdout.flush()
                    time.sleep(0.3) # 캡처 이모지 유지
                
                # 실제 마이크 볼륨 기반 동적 EQ 계산
                volume = 0.0
                with audio_buffer_lock:
                    if len(audio_buffer) > 0:
                        volume = np.max(np.abs(audio_buffer[-1]))
                
                # 시각화 (0~10칸)
                bars = int(min(volume * 100, 10))
                eq_str = "🟩" * bars + "⬛" * (10 - bars)
                emoji = spin_chars[spin_idx % len(spin_chars)]
                
                sys.stdout.write(f"\r{style(f'🎙️ {emoji} 관찰 중... [{eq_str}]', Color.CYAN)}".ljust(60))
                sys.stdout.flush()
                
                spin_idx += 1
                time.sleep(0.1)
                
            # 루프 종료 시 화면 클리어
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
                
            if not is_running:
                break
                
            exact_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print()
            print_styled(f"▶️ [{exact_timestamp}] [메인 루프] 국면 {chunk_idx}: {chunk_duration}초 분량 컷팅 및 백그라운드 스레드 전달...", Color.MAGENTA)
            
            # 음성 버퍼에서 10초 분량 가져오고 버퍼 비우기
            with audio_buffer_lock:
                if len(audio_buffer) > 0:
                    recording_data = np.concatenate(audio_buffer, axis=0)
                    cumulative_audio_data.append(recording_data) # 전체 오디오 데이터 누적
                    audio_buffer.clear()
                else:
                    recording_data = np.zeros((fs, 1), dtype='float32')
                    
            if not chunk_frames:
                chunk_frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
                    
            if len(recording_data) > fs * 1: # 1초 이상 녹음된 경우 분석 스레드 제출
                executor.submit(process_chunk, recording_data, chunk_frames, exact_timestamp, chunk_idx, fs, realtime_log_file, image_dir)
            
            chunk_idx += 1
            
    except KeyboardInterrupt:
        print()
        print_warning("[Ctrl+C 감지] 사용자에 의해 강제 종료됩니다.")
    finally:
        is_running = False # 루프가 끝났으므로 다른 스레드들에게도 확실히 종료 신호 전달
        print()
        print_success("(종료 루틴 진입) 마이크와 카메라 자원 회수 중...")
        cleanup_hardware() 
        print_info("진행 중인 백그라운드 분석 데이터(Gemini API 병합)가 있다면 완료될 때까지 잠시 대기합니다...")
        print_warning("👉 (만약 너무 오래 걸리면 Ctrl+C를 한 번 더 눌러 강제로 즉시 종료하세요)")
        
        try:
            executor.shutdown(wait=True, cancel_futures=False)
            # 최종 결과 종합 (로컬 변수가 선언되어 있을 때만)
            if session_id:
                generate_global_summary(session_id)
            # 종료 전 임시 파일 한번 더 비우기
            cleanup_temp_dir()
            print_styled("\n[완료] 윤이나의 관찰 및 분석이 모두 안전하게 종료되었습니다. 수고하셨습니다, 교수님!", Color.CYAN, Color.BOLD)
        except KeyboardInterrupt:
            print()
            print_error("🚨 강제 종료 신호 재입력 감지! 즉시 모든 프로세스를 처형하고 시스템을 탈출합니다.")
            os._exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch Presentation CLI Tool (Yoon Ina)")
    parser.add_argument("--chunk", type=int, default=10, help="Chunk duration in seconds (기본값: 10초)")
    parser.add_argument("--focus", type=str, default="", help="집중 분석할 키워드 또는 관점")
    args = parser.parse_args()
    
    main_loop(chunk_duration=args.chunk, focus_area=args.focus)
