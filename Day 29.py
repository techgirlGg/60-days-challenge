from openai import OpenAI
import os

# Load API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(prompt_text):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt_text
        )

        return response.output_text

    except Exception as e:
        return f"Error: {str(e)}"


# Test prompt
user_prompt = input("Ask something: ")

reply = generate_response(user_prompt)

print("\nAI Response:")
print(reply)