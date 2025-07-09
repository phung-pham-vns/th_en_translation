import json
import pandas as pd


# Load the JSON data from file
file_paths = {
    "nllb-200-distilled-600M": "dataset/nllb-200-distilled-600M.json",
    "opus_mt_th_en": "dataset/opus_mt_th_en.json",
    "gemma-3-4b-it-Q4_K_M": "dataset/gemma-3-4b-it-Q4_K_M.json",
    "gemma-3n-e4b_Q4_K_M": "dataset/gemma-3n-e4b_Q4_K_M.json",
    "gemma-3-4b-it-QAT-Q4_0": "dataset/gemma-3-4b-it-QAT-Q4_0.json",
    "gemini-2.5-flash-lite-preview-06-17": "dataset/gemini-2.5-flash-lite-preview-06-17.json",
    "gemini-2.5-pro": "dataset/gemini-2.5-pro.json",
}

stats = []

for model_name, file_path in file_paths.items():
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        continue

    total_time = 0.0
    total_llm_score = 0.0
    total_bleu = 0.0
    count = 0

    for item in data:
        total_time += item.get("time_second", 0)
        llm_score = (
            item.get("metric", {}).get("LLM-as-a-judge", {}).get("score", 0) or 0
        )
        total_llm_score += llm_score
        bleu = item.get("metric", {}).get("BLEU", {}).get("score", 0) or 0
        total_bleu += bleu
        count += 1

    average_time = total_time / count if count else 0
    average_llm_score = total_llm_score / count if count else 0
    average_bleu = total_bleu / count if count else 0

    stats.append(
        {
            "Model": model_name,
            "Samples": count,
            "Avg Time (s)": round(average_time, 4),
            "Avg LLM Score": round(average_llm_score, 4),
            "Avg BLEU Score": round(average_bleu, 4),
        }
    )

# Convert to DataFrame
df = pd.DataFrame(stats)

# Format as markdown table
markdown = "# 📊 Translation Model Performance Comparison\n\n"
markdown += df.to_markdown(index=False)

# Save to file
output_path = "dataset/performance.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown)
