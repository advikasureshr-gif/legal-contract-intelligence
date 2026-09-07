import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class NLLBTranslator:
    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    LANGUAGE_CODES = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "ta": "tam_Taml",
    }

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading translation model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            src_lang="eng_Latn",
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)

        self.model.eval()

        print("Translation model loaded.")

    def translate(self, text, target_language):
        """Translate a single piece of text."""
        return self.translate_many(
            [text],
            target_language,
            batch_size=1,
        )[0]

    def translate_many(self, texts, target_language, batch_size=4):
        """Translate multiple texts in small batches."""
        if target_language == "en":
            return texts

        if target_language not in self.LANGUAGE_CODES:
            raise ValueError(
                f"Unsupported target language: {target_language}"
            )

        if not texts:
            return []

        target_code = self.LANGUAGE_CODES[target_language]

        results = []

        for start in range(0, len(texts), batch_size):
            batch = [
                str(text) if text else ""
                for text in texts[start:start + batch_size]
            ]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.inference_mode():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        target_code
                    ),
                    max_length=512,
                    num_beams=3,
                )

            translated = self.tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True,
            )

            results.extend(translated)

        return results