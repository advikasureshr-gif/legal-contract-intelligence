from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.7-flash",
    contents="In one sentence, explain what an indemnification clause is."
)

print(response.text)