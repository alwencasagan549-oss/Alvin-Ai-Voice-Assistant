import asyncio
from app.llm import llm_service, build_messages, MessageParam
from app.config import settings

# Check what the messages look like
import base64
with open('image.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

messages = build_messages(
    [], 
    'What can you see in this image?', 
    max_history=1,
    system_prompt='You are Alvin, a helpful voice assistant with camera access. Describe what you see in the image accurately.',
    image_b64=b64
)

print('Messages:')
import json
for i, msg in enumerate(messages):
    if isinstance(msg.get('content'), list):
        print(f'  {i}: {msg["role"]} - content: [')
        for item in msg['content']:
            if item['type'] == 'text':
                print(f'    {{type: text, text: "{item["text"][:50]}..."}}')
            elif item['type'] == 'image_url':
                print(f'    {{type: image_url, image_url: {{url: "{item["image_url"]["url"][:50]}..."}}}}')
        print('  ]')
    else:
        print(f'  {i}: {msg["role"]} - {msg.get("content", "")[:100]}')