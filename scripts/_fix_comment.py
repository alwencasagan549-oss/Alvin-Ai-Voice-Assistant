#!/usr/bin/env python
"""One-off: fix the stale comment in app/llm.py _stream_with_tools."""
import sys

path = "app/llm.py"
with open(path, "rb") as f:
    data = f.read()

old = (
    b'                        # Not a question (or explicitly "don\'t search") \xe2\x80\x94\n'
    b'                        # do not offer the tool. Round-0 content is still\n'
    b'                        # buffered and narration-filtered below, so any\n'
    b'                        # hallucinated "let me search" preamble is still dropped.\n'
    b'                        current_tool_choice = "none"\n'
    b'                        round_tools = None'
)

new = (
    b'                        # Not a question (or explicitly "don\'t search") \xe2\x80\x94 do not\n'
    b'                        # offer the tool. With round_tools = None, no tool call is\n'
    b'                        # possible, so buffer_content is False below: round-0\n'
    b'                        # content streams normally and is still narration-filtered\n'
    b'                        # inline, catching any hallucinated "let me search" preamble.\n'
    b'                        current_tool_choice = "none"\n'
    b'                        round_tools = None'
)

if old not in data:
    print("ERROR: old string not found in file")
    sys.exit(1)

count = data.count(old)
data = data.replace(old, new)
with open(path, "wb") as f:
    f.write(data)
print(f"Replacement succeeded ({count} occurrence(s)).")
