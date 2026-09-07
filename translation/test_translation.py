from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    src_lang="eng_Latn"
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "This agreement shall remain in effect for five years."

inputs = tokenizer(text, return_tensors="pt")

translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml"),
    max_length=128
)

translation = tokenizer.batch_decode(
    translated_tokens,
    skip_special_tokens=True
)[0]

print("English:", text)
print("Hindi:", translation)