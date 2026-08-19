"""
Diagnostic: calls the same judge LLM used in run_ragas.py directly,
outside of RAGAS's evaluate(), to see whether it's the LLM call itself
that's hanging/failing, or something specific to how RAGAS drives it.

Usage:
    uv run python eval/debug_judge_llm.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

print("Creating ChatGoogleGenerativeAI client...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"]
)

print("Sending a trivial test prompt, timing it...")
start = time.time()
try:
    response = llm.invoke("Reply with exactly one word: OK")
    elapsed = time.time() - start
    print(f"SUCCESS in {elapsed:.1f}s. Response: {response.content!r}")
except Exception as e:
    elapsed = time.time() - start
    print(f"FAILED after {elapsed:.1f}s. Exception type: {type(e).__name__}")
    print(f"Exception: {e}")
