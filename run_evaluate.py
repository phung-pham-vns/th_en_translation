import os
import json
import time
import argparse
import evaluate
from dotenv import load_dotenv
from google import genai
from google.genai import types
import openai


def call_gemini(client: genai.Client, system_prompt: str, user_prompt: str) -> str:
    """Function to call Gemini API."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[system_prompt, user_prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
        ),
    )
    return response.text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thai-English translation evaluator using LLMs."
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="LLM provider",
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the JSON dataset"
    )
    parser.add_argument(
        "--openai_model", type=str, default="gpt-4o", help="OpenAI model name"
    )
    parser.add_argument(
        "--gemini_model", type=str, default="gemini-2.5-flash", help="Gemini model name"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep time between API calls (to avoid rate limit)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    # API setup
    openai.api_key = os.environ["OPENAI_API_KEY"]
    gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # System + user prompt templates
    system_prompt = (
        "You are a bilingual Thai-English translation evaluator. "
        "Your job is to evaluate the quality of a machine-generated English translation "
        "based on the original Thai sentence and a correct human reference translation."
    )

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
"""

    metric = evaluate.load("sacrebleu")

    with open(file=args.file_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        print("**" * 30)
        print(f"Evaluating sample {i + 1}/{len(data)}...")

        user_prompt = user_prompt_template.format(
            thai=item["thai"],
            English=item["english"],
            predict=item["predict"],
            time_second=item["time_second"],
        )

        try:
            if args.provider == "gemini":
                raw_output = call_gemini(gemini_client, system_prompt, user_prompt)
            else:
                response = openai.ChatCompletion.create(
                    model=args.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                raw_output = response["choices"][0]["message"]["content"]

            print("Raw output:", raw_output)

            json_start = raw_output.find("{")
            json_end = raw_output.rfind("}") + 1
            parsed_json = json.loads(raw_output[json_start:json_end])

            if "metric" not in item:
                item["metric"] = dict()
            # if "LLM-as-a-judge" not in item["metric"]:
            item["metric"]["LLM-as-a-judge"] = parsed_json
            # if "BLEU" not in item["metric"]:
            item["metric"]["BLEU"] = {
                "score": metric.compute(
                    predictions=[item["predict"]],
                    references=[[item["english"]]],
                )["score"]
                / 100.0
            }

        except Exception as e:
            print(f"Error evaluating sample {i + 1}: {e}")
            if "metric" not in item:
                item["metric"] = dict()
            item["metric"]["LLM-as-a-judge"] = {"score": None, "explanation": str(e)}
            item["metric"]["BLEU"] = {"score": None}

        time.sleep(args.sleep)

    with open(args.file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ Evaluation completed and saved to {args.file_path}")


if __name__ == "__main__":
    main()
