import re

with open("app/config.py", mode="r") as f:
    content = f.read()

# STT optimizations
content = content.replace("1.8)", "1.5)")
content = content.replace("1.2)", "1.0)")
content = content.replace("0.15)", "0.12)")

# Local fallback model optimization
content = content.replace('"small.en"', '"base.en"')

# TTS optimizations
content = content.replace(", 2)", ", 3)")
content = content.replace("-50.0)", "-45.0)")
content = content.replace(", 200)", ", 100)")
content = content.replace(", 120)", ", 80)")

with open("app/config.py", mode="w") as f:
    f.write(content)

print("Config optimizations applied successfully!")
