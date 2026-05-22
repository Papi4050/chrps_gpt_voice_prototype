# CHRPS GPT Voice Prototype

This project is a Python prototype for voice-first interaction with OpenAI models.

It includes:
- A simple text-only API smoke test.
- A voice chat loop (record -> transcribe -> respond -> optional speech output).
- A tool-calling example where the model can trigger local actions (`enable_teach`).

## How It Works

The main voice flow is implemented in `voice_chat.py`:

1. Record microphone audio to `input.wav` using `sounddevice` + `soundfile`.
2. Send that audio to OpenAI Speech-to-Text (`gpt-4o-transcribe`).
3. Send the transcribed text to a chat model (`gpt-4o`) with tool schemas.
4. If the model asks for a tool call:
   - Run the local Python handler (`handle_tool_call`).
   - Append tool output back into chat history.
   - Ask the model for a final natural-language response.
5. Optionally synthesize the assistant reply to speech (`gpt-4o-mini-tts`) and save as `reply.mp3`.

Conversation state is kept in an in-memory `history` list for multi-turn context.

## Files

- `voice_chat.py`: end-to-end voice chat + tool-calling loop.
- `actions_chat.py`: text-only demo of the same tool-calling pattern.
- `main.py`: minimal OpenAI API smoke test (`responses.create`).
- `list_mics.py`: prints available audio input/output devices.
- `environment.yml`: Conda environment definition.

## Requirements

- Python 3.12 (as defined in `environment.yml`)
- OpenAI API key
- Working microphone

## Setup

### 1) Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate gpt_voice_prototype
```

### 2) Add environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run

### Check microphones

```bash
python list_mics.py
```

If needed, set `MIC_INDEX` in `voice_chat.py` to the correct device index.

### Minimal API test

```bash
python main.py
```

### Voice chat + tools

```bash
python voice_chat.py
```

Usage in the prompt:
- Press `ENTER` to record for the default duration.
- Type a number (seconds) to override recording duration for one turn.
- Type `quit` to exit.

## Notes and Limitations

- `voice_chat.py` currently uses `os.startfile(...)` to open `reply.mp3`, which is Windows-specific.
  - On macOS, audio is still saved to `reply.mp3`; play it manually (or swap to `open reply.mp3`).
- Tool execution is intentionally whitelisted in `handle_tool_call` for safety.
- Generated files (`input.wav`, `reply.mp3`) are overwritten each turn.

## Extending This Prototype

- Add more tool schemas in `TOOLS` and implement handlers in `handle_tool_call`.
- Replace the placeholder `enable_teach` behavior with your real local workflow.
- Add robust error handling for network/audio failures.
- Add a platform-aware audio playback helper for macOS/Linux/Windows.
