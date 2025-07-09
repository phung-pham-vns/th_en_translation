import json
import time
import torch
import argparse
from transformers import pipeline


def load_pipeline(model_name: str, dtype: torch.dtype = torch.bfloat16):
    return pipeline(
        task="translation",
        model=model_name,
        torch_dtype=dtype,
    )


def th_to_en_translator(pipeline_func, text: str) -> str:
    result = pipeline_func(
        text,
        src_lang="tha_Latn",
        tgt_lang="eng_Latn",
    )
    return result[0]["translation_text"]


def main():
    parser = argparse.ArgumentParser(
        description="Thai-to-English translation using NLLB"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save translated output JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Translation model name",
    )
    args = parser.parse_args()

    # Load model pipeline
    translator = load_pipeline(args.model)

    # Load input data
    with open(args.input, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    for i, sample in enumerate(data):
        print(f"\n🔄 Translating {i+1}/{len(data)}")
        t1 = time.time()
        try:
            predict = th_to_en_translator(translator, sample["thai"])
        except Exception as e:
            predict = f"[ERROR] {str(e)}"
        t2 = time.time()

        sample["predict"] = predict
        sample["time_second"] = round(t2 - t1, 3)

        print(f"🇹🇭 Thai: {sample['thai']}")
        print(f"🇬🇧 English: {predict}")
        print(f"⏱ Time: {sample['time_second']}s")

    output_path = args.output or f"dataset/{args.model.split('/')[-1]}.json"
    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Translation completed and saved to: {output_path}")


if __name__ == "__main__":
    main()
