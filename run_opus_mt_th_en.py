from typing import Optional, Union, Generator
from transformers import MarianMTModel, MarianTokenizer


def chunks(lst: list, size: Optional[int] = None) -> Generator:
    """Yield successive chunks of a list."""
    if size is None or size <= 0:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i : i + size]


class ThToEnTranslator:
    def __init__(self, model_name_or_path: str = "Helsinki-NLP/opus-mt-th-en"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name_or_path)
        self.model = MarianMTModel.from_pretrained(model_name_or_path)

    def __call__(
        self, texts: Union[str, list[str]], batch_size: int = 1
    ) -> Union[str, list[str]]:
        """
        Translate Thai text(s) to English.
        Supports a single string or list of strings.
        Automatically batches for efficient translation.
        """
        if isinstance(texts, str):
            texts = [texts]

        translations = []
        for batch in chunks(texts, size=batch_size):
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self.model.generate(**inputs)
            translations.extend(
                [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            )
        return translations if len(translations) > 1 else translations[0]


if __name__ == "__main__":
    import json
    import time
    import argparse

    parser = argparse.ArgumentParser(
        description="Translate Thai to English using Helsinki-NLP/opus-mt-th-en."
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="dataset/opus_mt_th_en.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for translation"
    )
    args = parser.parse_args()

    translator = ThToEnTranslator("Helsinki-NLP/opus-mt-th-en")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, sample in enumerate(data):
        print(f"\n🔄 Translating {i+1}/{len(data)}")
        t1 = time.time()
        prediction = translator(sample["thai"], batch_size=args.batch_size)
        t2 = time.time()

        sample["predict"] = prediction
        sample["time_second"] = round(t2 - t1, 3)

        print(f"🇹🇭 Thai: {sample['thai']}")
        print(f"🇬🇧 English: {prediction}")
        print(f"⏱ Time: {sample['time_second']}s")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Translation completed and saved to: {args.output}")
