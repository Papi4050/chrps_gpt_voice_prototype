import os, sys, time
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
import json

# === CONFIG ===
SAMPLE_RATE = 16000
CHANNELS = 1
MIC_INDEX = None        # set to an int (see sd.query_devices()) or leave None for default
RECORD_SECONDS = 6      # simple fixed-duration push-to-talk
USE_TTS = True          # set False if you don't want spoken replies
VOICE = "alloy"         # pick any available TTS voice

# === INIT ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----- Tool schema(s) -----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "enable_teach",
            "description": "Enable teaching mode so the user can demonstrate behaviors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional short label for what the user wants to teach."
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

# Local state you might toggle from tools
STATE = {"teaching_enabled": False, "teaching_topic": None}

# ----- Chat history -----
history = [
    {"role": "system", "content":
        "You are a concise, helpful assistant. "
        "If the user indicates they want to teach you something, call the enable_teach function "
        "with an optional 'topic'. After calling a tool, continue to respond naturally."
    }
]

def record_wav(path: str, seconds: int):
    """Record from the chosen microphone into a WAV file."""
    print(f"\n🎙️  Recording {seconds}s... (mic={MIC_INDEX})")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=MIC_INDEX
    )
    sd.wait()
    sf.write(path, audio, SAMPLE_RATE)
    print("✅ Saved:", path)

def transcribe(path: str) -> str:
    """Send the WAV to OpenAI STT."""
    tr = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=open(path, "rb"),
    )
    return tr.text

def handle_tool_call(name: str, arguments_json: str) -> str:
    """Run a safe, whitelisted local action for the tool call."""
    if name == "enable_teach":
        print(f"[TOOL] Executing enable_teach with args={arguments_json}")
        args = json.loads(arguments_json or "{}")
        topic = args.get("topic") or "unspecified"
        # Example: toggle local state (replace with your real command if needed)
        STATE["teaching_enabled"] = True
        STATE["teaching_topic"] = topic

        # If you need to run an external command, do it safely (no shell=True)
        # import subprocess
        # result = subprocess.run(["enable_teach", "--topic", topic], capture_output=True, text=True)
        # output = result.stdout.strip() or result.stderr.strip() or "No output."
        output = f"Teaching mode enabled (topic='{topic}')."
        return output

    return "Tool not implemented."

def chat_with_actions(user_text: str) -> str:
    """One conversational turn with tool-calling support."""
    history.append({"role": "user", "content": user_text})

    # First pass: allow the model to call tools
    first = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
        tools=TOOLS,
        tool_choice="auto"
    )

    msg = first.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        print("\n[TOOL] Model requested tool(s):")  # <--- add
        # Append the assistant message (which contains tool call metadata)
        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls
        })

        # Execute each tool call and append its result
        for call in tool_calls:
            print(f"      - {call.function.name}({call.function.arguments})")  # <--- add
            name = call.function.name
            args = call.function.arguments or "{}"
            tool_result = handle_tool_call(name, args)

            print(f"[TOOL] {name} RESULT -> {tool_result}")  # <--- add
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": tool_result
            })

        # Second pass: have the model produce the final natural reply
        second = client.chat.completions.create(
            model="gpt-4o",
            messages=history
        )
        answer = second.choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        return answer

    # No tool call—just a normal reply
    answer = msg.content
    history.append({"role": "assistant", "content": answer})
    return answer

def speak(text: str, path: str = "reply.mp3"):
    # Text → speech (MP3 by default)
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    with open(path, "wb") as f:
        f.write(speech.read())

    # Open with default player on Windows
    try:
        os.startfile(path)  # Windows-only
    except Exception:
        print(f"(Audio saved to {path}. Open it to play.)")

def main():
    print("Voice chat + actions ready (your original audio playback kept).")
    while True:
        cmd = input("\nPress ENTER to talk (or type duration in seconds, or 'quit'): ").strip().lower()
        if cmd == "quit":
            break
        try:
            secs = int(cmd) if cmd else RECORD_SECONDS
        except ValueError:
            secs = RECORD_SECONDS

        wav_in = "input.wav"
        record_wav(wav_in, secs)
        print("📝 Transcribing...")
        user_text = transcribe(wav_in)
        print("You said:", user_text)

        print("🤖 Thinking...")
        answer = chat_with_actions(user_text)
        print("Assistant:", answer)

        if USE_TTS:
            print("🔊 Speaking...")
            speak(answer)

if __name__ == "__main__":
    main()
