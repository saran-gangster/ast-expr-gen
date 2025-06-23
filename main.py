import asyncio
import json
import math
import os
import random
import re
import aiohttp
from decimal import Decimal, ROUND_HALF_UP

import generators


def load_config():
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create it based on the README.")
        exit()
    except json.JSONDecodeError:
        print("Error: config.json is not a valid JSON file.")
        exit()

config = load_config()
TOKEN = config.get("DISCORD_TOKEN")
USERID = config.get("USER_ID")
CHANNELID = config.get("CHANNEL_ID")
COUNTING_BOT_ID = config.get("COUNTING_BOT_ID")
STATE_FILE = config.get("STATE_FILE", "bot_state.json")
GENERATOR_MODE = config.get("GENERATOR_MODE", "AST")
HEADERS = {"authorization": TOKEN, "content-type": "application/json"}
last_processed_message_id = None


_FIB_CACHE = {0: 0, 1: 1}
def _fibonacci(n):
    n = int(round_half_up(n))
    if n in _FIB_CACHE: return _FIB_CACHE[n]
    if n > 100 or n < -100: raise ValueError("Fibonacci input out of range [-100, 100]")
    if n > 0: val = _fibonacci(n - 1) + _fibonacci(n - 2)
    else: val = _fibonacci(n + 2) - _fibonacci(n + 1)
    _FIB_CACHE[n] = val
    return val

def round_half_up(n):
    """Rounds a number to the nearest integer, with halves rounded up (As per the Counting bot Docs)"""
    if not isinstance(n, (int, float)) or not math.isfinite(n):
        return 0 # forces retry by a non target value
    return int(Decimal(str(n)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))

def _bot_log_quirk(value, base):
    """
    Handles the bot's evaluation order for logs.
    AST generator produces log(value, base), which is the standard way for math functions.
    Python's math.log is log(value, base), so this wrapper ensures correctness.
    """
    return math.log(value, base)

SAFE_GLOBALS = {
    "__builtins__": None, "pi": math.pi, "e": math.e, "tau": math.tau, "phi": (1 + math.sqrt(5)) / 2, "γ": 0.5772156649,
    "abs": abs, "floor": math.floor, "ceil": math.ceil, "round": round_half_up, "trunc": math.trunc,
    "gcd": math.gcd, "lcm": math.lcm, "fact": math.factorial, "factorial": math.factorial, "fibonacci": _fibonacci,
    "gamma": math.gamma, "lgamma": math.lgamma, "pow": pow, "sqrt": math.sqrt,
    "cbrt": lambda x: x**(1/3) if x >= 0 else -(-x)**(1/3),
    "exp": math.exp, "exp2": math.exp2, "expm1": math.expm1,
    "log": _bot_log_quirk, "ln": math.log, "log2": math.log2, "log10": math.log10, "log1p": math.log1p,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "cot": lambda x: 1/math.tan(x),
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "rad": math.radians, "deg": math.degrees, "hypot": math.hypot
}

def prepare_expression(expr: str) -> str:
    expr = expr.strip().lower().replace(" ", "").replace("_", "")
    if re.fullmatch(r"0\d+", expr): return str(int(expr))
    replacements = {"^": "**", "⊻": "^", "×": "*", "÷": "/", "−": "-", "√": "sqrt"}
    for old, new in replacements.items(): expr = expr.replace(old, new)
    return expr

def evaluate_expression(expr: str):
    if not expr: raise ValueError("Empty expression.")
    prepared_expr = prepare_expression(expr)
    try:
        result = eval(prepared_expr, SAFE_GLOBALS, {})
        if isinstance(result, (int, float)) and math.isfinite(result): return result
        raise ValueError(f"Non-numeric or non-finite result: {result}")
    except Exception as e:
        raise ValueError(f"Evaluation failed for '{prepared_expr}': {e}")


def load_state():
    if not os.path.exists(STATE_FILE): return None
    try:
        with open(STATE_FILE, "r") as f: return json.load(f).get("last_processed_message_id")
    except (json.JSONDecodeError, AttributeError): return None

def save_state(message_id):
    with open(STATE_FILE, "w") as f: json.dump({"last_processed_message_id": message_id}, f)

async def get_latest_valid_message(session, last_id):
    url = f"https://discord.com/api/v9/channels/{CHANNELID}/messages?limit=10"
    try:
        async with session.get(url, headers=HEADERS) as response:
            response.raise_for_status()
            messages = await response.json()
            for msg in messages:
                if msg['id'] == last_id: return None 
                if msg['author']['id'] == USERID: continue
                content = msg.get('content', '')
                if not content: continue
                if msg['author']['id'] == COUNTING_BOT_ID and "ruined it" in content.lower():
                    print("Detected a ruined count. Next number is 1.")
                    return {'value': 0, 'message_id': msg['id']}
                try:
                    final_number = round_half_up(evaluate_expression(content))
                    print(f"Found valid message: '{content}' (Rounded: {final_number}) by {msg['author']['username']}")
                    return {'value': final_number, 'message_id': msg['id']}
                except ValueError: continue
            return None
    except aiohttp.ClientError as e:
        print(f"Error fetching messages: {e}")
        return None

async def send_message(session, message_content):
    url = f"https://discord.com/api/v9/channels/{CHANNELID}/messages"
    payload = {"content": str(message_content)}
    try:
        async with session.post(url, headers=HEADERS, data=json.dumps(payload)) as response:
            if response.ok:
                print(f"Successfully sent: {str(message_content)[:100]}...")
                return (await response.json()).get('id')
            else:
                print(f"Failed to send message: {response.status} - {await response.text()}")
                return None
    except aiohttp.ClientError as e:
        print(f"Error sending message: {e}")
        return None

async def main_loop():
    global last_processed_message_id
    last_processed_message_id = load_state()
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                latest_message = await get_latest_valid_message(session, last_processed_message_id)
                if latest_message:
                    my_number = latest_message['value'] + 1
                    my_expression, _ = generators.generate_expression(
                        final_number=my_number, mode=GENERATOR_MODE,
                        eval_func=evaluate_expression, round_func=round_half_up
                    )
                    await asyncio.sleep(random.uniform(0.5, 1.2))
                    new_message_id = await send_message(session, my_expression)
                    last_processed_message_id = new_message_id or latest_message['message_id']
                    save_state(last_processed_message_id)
                else:
                    await asyncio.sleep(random.uniform(4, 8))
            except Exception as e:
                print(f"An unexpected error occurred in the main loop: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    if not all([TOKEN, USERID, CHANNELID, COUNTING_BOT_ID]):
        print("Error: Please complete your details in config.json before running.")
    else:
        try:
            print("Starting Discord counting bot...")
            print(f"Generator Mode: {GENERATOR_MODE}")
            asyncio.run(main_loop())
        except KeyboardInterrupt:
            print("\nBot stopped by user.")