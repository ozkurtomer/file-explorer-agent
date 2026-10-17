from ollama import chat

response = chat(
    model="qwen3:4b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Local model is ready."
        }
    ]
)

print(response.message.content)