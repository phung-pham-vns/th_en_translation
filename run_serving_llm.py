import json
import time
import requests
from pydantic import BaseModel
from typing import Literal, Optional
import argparse

# ========== Configuration ==========
LLAMA_API_URL = "http://localhost:1234/v1/chat/completions"


# ========== Prompt Templates ==========
SYSTEM_PROMPT = (
    "You are a professional translation expert specializing in Thai-to-English translation.\n\n"
    "Your task is to translate Thai queries into clear, natural, and grammatically correct English, while fully preserving the original meaning, tone, and intent.\n"
    "Maintain the original format — especially if the input is a question — and ensure contextual accuracy.\n\n"
    "You must strictly preserve the following fixed Thai terms by translating them exactly as shown:\n\n"
    "{\n"
    '  "ระยะเริ่มแทงยอด": "Initial Bud Stage",\n'
    '  "ระยะหางปla-ใบคลี่": "Fishbone-Curled Leaf Stage",\n'
    '  "ระยะใบเพสลาดอ่อน": "Initial Semi-Mature Leaf Stage",\n'
    '  "ระยะใบเพสลาด": "Semi-Mature Leaf Stage",\n'
    '  "ระยะใบแก่": "Mature Leaf Stage",\n'
    '  "ระยะไข่ปลา": "Early Bud Initiation Stage",\n'
    '  "ระยะตาปู": "Small Bud Stage",\n'
    '  "ระยะเหยียดตีนหนู": "Bud Elongation Stage",\n'
    '  "ระยะกระดุม": "Button Stage",\n'
    '  "ระยะมะเขือพวง": "Enlarged Bud Stage",\n'
    '  "ระยะหัวกำไล": "Pre-Flowering Stage",\n'
    '  "ระยะดอกขาว": "Flower Maturity Stage",\n'
    '  "ระยะดอกบาน": "Flowering Stage (Anthesis)",\n'
    '  "ระยะหางแย้ไหม้": "Post-Anthesis / Early Fruit Set Stage",\n'
    '  "ระยะไข่ไก่": "Small Fruit Stage",\n'
    '  "ระยะกระป๋องนม": "Medium Fruit Stage",\n'
    '  "ระยะขยายพู": "Maturing Fruit Stage",\n'
    '  "ระยะเริ่มสุกแก่-เก็บเกี่ยว": "Harvest Maturity Stage",\n'
    '  "โรครากเน่า โคนเน่า และผลเน่า": "Root rot, Foot rot, and Fruit rot",\n'
    '  "โรคแอนแทรคโนส": "Anthracnose",\n'
    '  "โรคใบจุดและผลเน่าที่เกิดจากเชื้อรา Phomopsis": "Leaf spot and fruit rot from Phomopsis",\n'
    '  "โรคผลเน่าที่เกิดจากเชื้อรา Lasiodiplodia": "Fruit rot from Lasiodiplodia",\n'
    '  "โรคราดำ": "Sooty mold",\n'
    '  "โรคใบติดและใบไหม้": "Rhizoctonia Leaf Fall, Rhizoctonia Leaf Blight",\n'
    '  "โรคใบจุดสาหร่ายหรือใบจุดสนิม": "Algal Leaf Spot",\n'
    '  "โรคกิ่งแห้ง": "Die-back",\n'
    '  "หนอนเจาะเมล็ดทุเรียน": "Durian Seed Borer",\n'
    '  "หนอนเจาะผล": "Yellow Peach Moth",\n'
    '  "เพลี้ยไก่แจ้ทุเรียน": "Durian Psyllid",\n'
    '  "เพลี้ยไฟพริก": "Chili Thrips",\n'
    '  "เพลี้ยแป้ง": "Mealybugs",\n'
    '  "ไรแดงแอฟริกัน": "African red mite",\n'
    '  "มอดเจาะลำต้น": "Shot Hole Borer",\n'
    '  "ออกดอก": "flowering",\n'
    '  "ดอกบาน": "blooming"\n'
    "}"
)

# Instruction template for the user query
USER_PROMPT_TEMPLATE = """Translate the following Thai query to English:
{thai_query}
Provide an accurate English translation that preserves the original meaning and intent.  
Return the translation in text.
```"""


# ========== Response Model ==========
class ChoiceMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class Choice(BaseModel):
    message: ChoiceMessage


class CompletionResponse(BaseModel):
    choices: list[Choice]


# ========== Translator Function ==========
def th_to_en_translator(
    text: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    model_name: str = "gemma-3-4b-it",
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(thai_query=text)},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(LLAMA_API_URL, json=payload)
        response.raise_for_status()
        parsed = CompletionResponse.parse_obj(response.json())
        return parsed.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"


# ========== Main Execution ==========
def main():
    parser = argparse.ArgumentParser(
        description="Thai-to-English translation via llama.cpp"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/evaluation.json",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-4b-it",
        help="Model name",
    )
    args = parser.parse_args()

    output_path = args.output or f"dataset/{args.model}.json"

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, sample in enumerate(data):
        print(f"\n🔄 Translating {i + 1}/{len(data)}")
        t1 = time.time()
        translation = th_to_en_translator(sample["thai"], model_name=args.model)
        t2 = time.time()

        sample["predict"] = translation
        sample["time_second"] = round(t2 - t1, 3)

        print(f"🇹🇭 Thai: {sample['thai']}")
        print(f"🇬🇧 Translated: {translation}")
        print(f"⏱ Time: {sample['time_second']}s")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Translations saved to: {output_path}")


if __name__ == "__main__":
    main()
