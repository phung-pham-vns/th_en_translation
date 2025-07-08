import requests
from pydantic import BaseModel


LLAMA_API_URL = "http://localhost:9422/v1/chat/completions"

system_prompt = (
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
default_instruction = """Translate the following Thai query to English:
{thai_query}
Provide an accurate English translation that preserves the original meaning and intent.  
Return the translation in text.
```"""


def th_to_en_translator(
    text: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    model_name: str = "gemma-3-4b-it",
) -> str:
    messages = [
        {
            "role": "system",
            # "content": "Translate the following Thai sentence into English in the formal way. Only output the English sentence. Do not explain.",
            # "content": "You are a professional translator. Translate Thai to English. Only output the translated English sentence. Do not explain.",
            "content": system_prompt,
        },
        {
            "role": "user",
            # "content": text,
            "content": default_instruction.format(thai_query=text),
        },
    ]

    payload = {
        "model": model_name,  # llama.cpp ignores this but it's required for OpenAI format
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(LLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        translated = result["choices"][0]["message"]["content"]
        return translated.strip()
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    import time

    model_name = "gemma-3-4b-it-new-prompt"

    with open(file="dataset/translation_testset.json", mode="r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        t1 = time.time()
        predict = th_to_en_translator(sample["thai"], model_name=model_name)
        t2 = time.time()
        sample["predict"] = predict
        sample["time_second"] = t2 - t1

        print("*" * 30)
        print(f"Thailand: {sample['thai']}")
        print(f"Translated: {predict}")
        print(f"Time: {t2 - t1}s")

    with open(file=f"dataset/{model_name}.json", mode="w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
