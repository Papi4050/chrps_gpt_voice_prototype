# hello_openai.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.responses.create(
    model="gpt-4o",            # or "gpt-4o-mini" for cheaper calls
    input="Write a one-sentence pep talk for a tired researcher."
)

print(resp.output_text)
