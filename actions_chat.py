# actions_chat.py
import os, json, subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) Define the tool schemas you want the model to be able to call
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "enable_teach",
            "description": "Enable teaching mode so the user can demonstrate behaviors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional short label for what the user wants to teach."}
                },
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

# 2) Local handler(s) for tool calls (NO shell=True; keep a strict whitelist)
def handle_tool_call(name: str, arguments_json: str) -> str:
    if name == "enable_teach":
        args = json.loads(arguments_json or "{}")
        topic = args.get("topic") or "unspecified"

        # Example A: call a Windows command (replace with your real command)
        # result = subprocess.run(["enable_teach", "--topic", topic], capture_output=True, text=True)
        # safe_result = result.stdout.strip() or result.stderr.strip() or "No output."

        # Example B: do it in Python (toggle a flag, start a thread, etc.)
        safe_result = f"Teaching mode enabled (topic='{topic}')."

        return safe_result

    # Unknown tool (shouldn’t happen if you keep a whitelist)
    return "Tool not implemented."

# 3) Chat loop with tool-calling
history = [
    {"role": "system", "content": (
        "You are a helpful assistant. If the user indicates they want to teach you something, "
        "call the enable_teach function with an optional 'topic'. After calling a tool, "
        "continue to respond naturally to the user."
    )}
]

def chat_once(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})

    # First pass: let the model decide whether to call a tool
    first = client.chat.completions.create(
        model="gpt-4o",
        messages=history,
        tools=TOOLS,
        tool_choice="auto"  # let the model call a tool when appropriate
    )

    msg = first.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        # Execute each tool call, append tool results, then ask the model to finish the reply
        history.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        for call in tool_calls:
            name = call.function.name
            args = call.function.arguments or "{}"
            tool_result = handle_tool_call(name, args)

            # Add the tool result so the model can see what happened
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": tool_result
            })

        # Second pass: model produces a natural-language response that reflects the tool outcome
        second = client.chat.completions.create(
            model="gpt-4o",
            messages=history
        )
        answer = second.choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        return answer

    else:
        # No tool call—just a normal reply
        answer = msg.content
        history.append({"role": "assistant", "content": answer})
        return answer

# --- Demo ---
if __name__ == "__main__":
    print(chat_once("I want to teach you how to tag items during labeling."))
    print(chat_once("Cool—what did you just do behind the scenes?"))
