import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser(description="Call the JSON tool with a specified model.")
parser.add_argument("--model", default="google/gemini-2.0-flash-001", help="Model provider, like 'google/gemini-2.0-flash-001'.")
args = parser.parse_args()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL = args.model
print(f"--- model = {MODEL} ---")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def get_current_weather(location):
    """Gets the current weather for a specified location."""
    print(f"--- Tool Function Called: get_current_weather(location='{location}') ---")
    weather_info = {"temperature": "above expectations", "unit": "of any human being", "description": "Grumpy with occasional rainbows"}
    return json.dumps(weather_info)

tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., Boston",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

user_prompt = "What's the weather in Boston?"
print(f"User: {user_prompt}")

messages = [{"role": "user", "content": user_prompt}]

print("--- Step 1: Sending request to OpenAI to see if a tool is needed ---")
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools_definition,
    tool_choice="auto"
)

response_message = response.choices[0].message
messages.append(response_message) 

if not response_message.tool_calls:
    print("--- Step 2: OpenAI responded directly (no tool call) ---")
    print(f"Assistant: {response_message.content}")
    exit(1)

print("--- Step 2: OpenAI requested a tool call ---")

tool_call = response_message.tool_calls[0]
function_name = tool_call.function.name
arguments_json = tool_call.function.arguments
arguments = json.loads(arguments_json)

print(f"   Function Name: {function_name}")
print(f"   Arguments: {arguments}")

function_response_content = get_current_weather(**arguments) 
print(f"   Function Result: {function_response_content}")

print("--- Step 3: Sending function result back to OpenAI ---")

messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": function_response_content,
    }
)

final_response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)

final_answer = final_response.choices[0].message.content
print("--- Step 4: Final Answer from OpenAI ---")
print(f"Assistant: {final_answer}")