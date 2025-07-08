import time
from run_llama_cpp import th_to_en_translator


if __name__ == "__main__":
    translator = th_to_en_translator()

    # Interactive translation loop
    print("🌐 Thai-to-English Translator (type 'goodbye' to exit)")
    while True:
        thai_text = input(
            "📝 Enter Thai text (use '|' to separate multiple sentences): "
        ).strip()
        if thai_text.lower() in ["goodbye", "exit", "quit"]:
            print("👋 Exiting translator.")
            break
        if not thai_text:
            continue

        try:
            start = time.time()
            if "|" in thai_text:
                thai_sentences = [s.strip() for s in thai_text.split("|") if s.strip()]
                translations = translator(thai_sentences, batch_size=4)
                print("🔁 Translations:")
                for i, (th, en) in enumerate(zip(thai_sentences, translations), 1):
                    print(f"  {i}. 🇹🇭 {th} → 🇬🇧 {en}")
            else:
                translation = translator(thai_text)
                print(f"🔁 Translation: {translation}")
            end = time.time()
            print(f"⏱ Took {end - start:.2f} seconds\n")
        except Exception as e:
            print(f"⚠️ Error: {e}")
