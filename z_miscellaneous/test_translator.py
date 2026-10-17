from translation.translator import NLLBTranslator


translator = NLLBTranslator()

text = "This agreement shall remain in effect for five years."

print("\nEnglish:")
print(text)

print("\nHindi:")
print(translator.translate(text, "hi"))

print("\nTamil:")
print(translator.translate(text, "ta"))