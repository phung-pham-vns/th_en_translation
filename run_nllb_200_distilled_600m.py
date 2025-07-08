import torch
from transformers import pipeline


translation_pipeline = pipeline(
    task="translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16,
)


def th_to_en_translator(text: str) -> str:
    texts = translation_pipeline(
        text,
        src_lang="tha_Latn",
        tgt_lang="eng_Latn",
    )
    return texts[0]["translation_text"]


if __name__ == "__main__":
    import json
    import time

    with open(file="dataset/translation_testset.json", mode="r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        t1 = time.time()
        predict = th_to_en_translator(sample["thai"])
        t2 = time.time()
        sample["predict"] = predict
        sample["time_second"] = t2 - t1

        print("*" * 30)
        print(f"Thailand: {sample['thai']}")
        print(f"Translated: {predict}")
        print(f"Time: {t2 - t1}s")

    with open(
        file="dataset/nllb-200-distilled-600M.json", mode="w", encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
