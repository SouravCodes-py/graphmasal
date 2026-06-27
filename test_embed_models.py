import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="test"
)
print(len(result["embedding"]))