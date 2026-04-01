"""
Tool call accuracy module — tests whether models correctly invoke functions,
pass the right arguments, and know when NOT to call a tool.
"""
import json
from ..services.llm_client import LLMClient

# ---------- test schemas ----------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database for records matching a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "table": {"type": "string", "description": "Table name"},
                    "limit": {"type": "number", "description": "Max results"},
                },
                "required": ["query", "table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
        },
    },
]

# ---------- test cases ----------

TEST_CASES = [
    {
        "user_message": "What's the weather like in Paris?",
        "expected_tool": "get_weather",
        "expected_args_contain": {"city": "Paris"},
        "should_call_tool": True,
    },
    {
        "user_message": "Search the users table for records matching 'John'",
        "expected_tool": "search_database",
        "expected_args_contain": {"query": "John", "table": "users"},
        "should_call_tool": True,
    },
    {
        "user_message": "Send an email to bob@example.com with subject 'Meeting' and body 'See you at 3pm'",
        "expected_tool": "send_email",
        "expected_args_contain": {"to": "bob@example.com"},
        "should_call_tool": True,
    },
    {
        "user_message": "What is 25 * 47?",
        "expected_tool": "calculate",
        "expected_args_contain": {},
        "should_call_tool": True,
    },
    {
        "user_message": "Tell me a joke about programming",
        "expected_tool": None,
        "expected_args_contain": {},
        "should_call_tool": False,
    },
    {
        "user_message": "What is the meaning of life?",
        "expected_tool": None,
        "expected_args_contain": {},
        "should_call_tool": False,
    },
    {
        "user_message": "What's the current temperature in Tokyo in celsius?",
        "expected_tool": "get_weather",
        "expected_args_contain": {"city": "Tokyo"},
        "should_call_tool": True,
    },
    {
        "user_message": "How are you today?",
        "expected_tool": None,
        "expected_args_contain": {},
        "should_call_tool": False,
    },
    {
        "user_message": "Find users in the orders table with query 'pending'",
        "expected_tool": "search_database",
        "expected_args_contain": {"table": "orders"},
        "should_call_tool": True,
    },
    {
        "user_message": "Calculate 15% of 2500",
        "expected_tool": "calculate",
        "expected_args_contain": {},
        "should_call_tool": True,
    },
]


def _check_args(actual_args: dict, expected_contain: dict) -> bool:
    """Check if actual args contain expected key-value pairs (case-insensitive for strings)."""
    for key, expected_val in expected_contain.items():
        actual_val = actual_args.get(key)
        if actual_val is None:
            return False
        if isinstance(expected_val, str) and isinstance(actual_val, str):
            if expected_val.lower() not in actual_val.lower():
                return False
        elif actual_val != expected_val:
            return False
    return True


async def run_tool_call_eval(client: LLMClient) -> dict:
    """Run tool call accuracy evaluation."""
    true_pos = 0   # correctly called the right tool
    true_neg = 0   # correctly did NOT call a tool
    false_pos = 0  # called a tool when it shouldn't have
    false_neg = 0  # didn't call a tool when it should have
    correct_args = 0
    total = len(TEST_CASES)
    details = []

    for tc in TEST_CASES:
        messages = [{"role": "user", "content": tc["user_message"]}]
        response = await client.chat(messages, tools=TOOL_DEFINITIONS)

        called_tool = response["tool_calls"] is not None and len(response["tool_calls"]) > 0
        result_detail = {
            "user_message": tc["user_message"],
            "expected_tool": tc["expected_tool"],
            "should_call_tool": tc["should_call_tool"],
            "called_tool": called_tool,
        }

        if tc["should_call_tool"]:
            if called_tool:
                actual_name = response["tool_calls"][0]["name"]
                actual_args = response["tool_calls"][0]["arguments"]
                name_match = actual_name == tc["expected_tool"]
                args_match = _check_args(actual_args, tc["expected_args_contain"])

                if name_match:
                    true_pos += 1
                    if args_match:
                        correct_args += 1
                else:
                    false_pos += 1  # called wrong tool

                result_detail["actual_tool"] = actual_name
                result_detail["actual_args"] = actual_args
                result_detail["name_correct"] = name_match
                result_detail["args_correct"] = args_match
            else:
                false_neg += 1
                result_detail["actual_tool"] = None
        else:
            if not called_tool:
                true_neg += 1
            else:
                false_pos += 1
                result_detail["actual_tool"] = response["tool_calls"][0]["name"] if response["tool_calls"] else None

        details.append(result_detail)

    total_predictions = true_pos + false_pos
    total_actual = true_pos + false_neg

    accuracy = (true_pos + true_neg) / total if total > 0 else 0
    precision = true_pos / total_predictions if total_predictions > 0 else 0
    recall = true_pos / total_actual if total_actual > 0 else 0
    fpr = false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "total_tests": total,
        "details": details,
    }
