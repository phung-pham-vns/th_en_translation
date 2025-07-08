from typing import Optional, Union, Generator
from transformers import MarianMTModel, MarianTokenizer


def chunks(lst: list, size: Optional[int] = None) -> Union[list, Generator]:
    if size is None:
        yield lst
    else:
        for i in range(0, len(lst), size):
            yield lst[i : i + size]


class ThToEnTranslator:
    def __init__(self, model_name_or_path: str = "Helsinki-NLP/opus-mt-th-en"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name_or_path)
        self.model = MarianMTModel.from_pretrained(model_name_or_path)

    def __call__(self, texts: str | list[str], batch_size: int = 1):
        """
        Translate Thai text(s) to English. Accepts a single string or a list of strings.
        Automatically batches input for efficient translation.
        """
        if isinstance(texts, str):
            texts = [texts]

        translations = []
        for test_chunk in chunks(texts, size=batch_size):
            inputs = self.tokenizer(
                test_chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            outputs = self.model.generate(**inputs)
            translations.extend(
                [self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            )
        return translations if len(translations) > 1 else translations[0]


if __name__ == "__main__":
    import json~
    import time

    th_to_en_translator = ThToEnTranslator(
        model_name_or_path="Helsinki-NLP/opus-mt-th-en"
    )

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

    with open(file="dataset/opus_mt_th_en.json", mode="w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
