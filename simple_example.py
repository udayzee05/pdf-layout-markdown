"""
Simple Example: PDF to Markdown to LLM
========================================
This is a minimal example showing how to:
1. Convert a PDF to Markdown using pdf-layout-markdown
2. Pass the Markdown to an LLM for analysis
"""

from pdf_layout_markdown import PDFLayoutAnalyzer
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Step 1: Convert PDF to Markdown
print("Step 1: Converting PDF to Markdown...")
analyzer = PDFLayoutAnalyzer("invoice.pdf")
markdown_content = analyzer.convert()

# Save the markdown
with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)
print("✅ Markdown saved to output.md")

# Step 2: Pass Markdown to LLM
print("\nStep 2: Analyzing with LLM...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a document analyst. Extract key information from the provided document and return it as JSON."
        },
        {
            "role": "user",
            "content": f"Analyze this document and extract key information:\n\n{markdown_content}"
        }
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

# Step 3: Get the result
result = json.loads(response.choices[0].message.content)

# Save the result
with open("analysis_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("✅ Analysis saved to analysis_result.json")
print(f"\nTokens used: {response.usage.total_tokens}")
print(f"Model: {response.model}")

# Print a preview of the result
print("\n" + "="*60)
print("ANALYSIS PREVIEW:")
print("="*60)
print(json.dumps(result, indent=2)[:500] + "...")
