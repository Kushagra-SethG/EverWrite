import google.genai as genai
from config import GEMINI_API_KEY, MODEL_NAME

client = genai.Client()

def generate_response(prompt):
    response = client.models.generate_content(
        config={
            "temperature": 0.7,
            "maxOutputTokens": 1500,
            "system_instruction": prompt
        },
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


def generate_response_stream(prompt):
    """Yields text chunks as the model streams them."""
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.7,
            "maxOutputTokens": 1500,
            "system_instruction": prompt
        }
    ):
        if chunk.text:
            yield chunk.text