import os
import openai
import json
import time
import evaluate
from dotenv import load_dotenv

load_dotenv()

# 🔐 Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]  # Replace or use environment variable

# ✅ System prompt (sets evaluator behavior)
system_prompt = (
    "You are a bilingual Thai-English translation evaluator. "
    "Your job is to evaluate the quality of a machine-generated English translation "
    "based on the original Thai sentence and a correct human reference translation."
)

# ✅ User prompt template
user_prompt_template = """Evaluate the following translation:

Thai (source):
{thai}

Ground Truth (reference translation):
{English}

Predicted Translation (model output):
{predict}

Time taken: {time_second} seconds

Instructions:
1. Compare the model prediction with the ground truth.
2. Assign a score between 0 (completely wrong) and 1 (perfect match), allowing intermediate values like 0.6, 0.85, etc.
3. Consider meaning preservation, fluency, and correctness of terminology.
4. Return your evaluation in this JSON format:

```json
{{
  "score": <float between 0 and 1>,
  "explanation": "<brief explanation>"
}}
```"""


metric = evaluate.load("sacrebleu")

file_path = "/Users/mac/Documents/PHUNGPX/th_en_translation/dataset/nllb-200-distilled-600M.json"

with open(file=file_path, mode="r", encoding="utf-8") as f:
    data = json.load(f)


for i, item in enumerate(data):
    print(f"Evaluating sample {i + 1}/{len(data)}...")
    print(item)
    user_prompt = user_prompt_template.format(
        thai=item["thai"],
        English=item["english"],
        predict=item["predict"],
        time_second=item["time_second"],
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        # Extract and parse the JSON from the response
        raw_output = response["choices"][0]["message"]["content"]
        print("Raw output:", raw_output)

        # Try to safely parse JSON block from response
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        parsed_json = json.loads(raw_output[json_start:json_end])

        # Store results
        if "metric" not in item:
            item["metric"] = dict()
        if "LLM-as-a-judge" not in item["metric"]:
            item["metric"]["LLM-as-a-judge"] = parsed_json
        if "BLEU" not in item["metric"]:
            item["metric"]["BLEU"] = {
                "score": metric.compute(
                    predictions=[item["predict"]], references=[[item["english"]]]
                )["score"]
                / 100.0
            }

    except Exception as e:
        print(f"Error evaluating sample {i + 1}: {e}")
        if "metric" not in item:
            item["metric"] = dict()
        item["metric"]["LLM-as-a-judge"] = {"score": None, "explanation": str(e)}
        item["metric"]["BLEU"] = {"score": None}

    time.sleep(1)  # Optional: avoid rate limits

# ✅ Save results
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ Evaluation completed and saved to {file_path}")
