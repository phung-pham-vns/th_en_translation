import os
import json
import time
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load .env and initialize Gemini client
def setup_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set in environment.")
    return genai.Client(api_key=api_key)


# Fixed translation terms
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

# Instruction template
user_prompt_template = """Translate the following Thai query to English:
{thai_query}
Provide an accurate English translation that preserves the original meaning and intent.  
Return the translation in text.
```"""


def translate(text: str, model: str, client: genai.Client) -> str:
    user_prompt = user_prompt_template.format(thai_query=text)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[system_prompt, user_prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            ),
        )
        return response.text
    except Exception as e:
        return f"[ERROR] {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Thai-to-English translation using Gemini"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model to use",
    )

    args = parser.parse_args()

    client = setup_gemini_client()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, sample in enumerate(data):
        print(f"\n🔄 Processing sample {i+1}/{len(data)}")
        t_start = time.time()
        prediction = translate(sample["thai"], model=args.model, client=client)
        t_end = time.time()

        sample["predict"] = prediction
        sample["time_second"] = round(t_end - t_start, 3)

        print(f"TH: {sample['thai']}")
        print(f"EN: {prediction}")
        print(f"⏱ Time: {sample['time_second']}s")

    out_path = args.output or f"dataset/{args.model.replace('/', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
