import json
import os
from dotenv import load_dotenv

# Ensure OPENAI_API_KEY is loaded
load_dotenv()

from agents.knowledge_extraction import KnowledgeExtractionAgent

def main():
    agent = KnowledgeExtractionAgent()
    
    sample_text = """
    Functions are mappings from one set to another.
    Composite Functions combine multiple functions.
    The Chain Rule is used when differentiating composite functions.
    """
    
    print("Extracting from sample text...")
    result = agent.process_text(sample_text)
    
    print("\nExtraction Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
