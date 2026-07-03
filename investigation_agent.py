import os
import json
import re
import pandas as pd
from groq import Groq, RateLimitError, APIStatusError

# Import tool definitions and backend implementations
from agent_tools import (
    TOOL_DEFINITIONS,
    tool_get_account_history,
    tool_get_pattern_deviation,
    tool_get_model_score,
    _load_data
)

def convert_anthropic_to_openai_tools(anthropic_tools):
    """
    Translates tool definitions from Anthropic format (name, description, input_schema)
    into OpenAI / Groq standard function-calling format.
    """
    openai_tools = []
    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("input_schema", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
        })
    return openai_tools

def _extract_json(text):
    """
    Helper function to robustly extract and parse JSON from the model's text response.
    Supports raw JSON, markdown-fenced JSON blocks, or extracting the first nested structure.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for JSON within markdown code blocks (e.g. ```json ... ```)
    markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find the outer-most curly braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON structure found in the response.")

def investigate_transaction(transaction_id, account_id, before_time):
    """
    Runs a ReAct-style agent loop using the Groq API to analyze a target transaction's fraud risk.
    Gathers evidence using tools and returns a structured risk decision.

    Parameters:
    -----------
    transaction_id : str
        The unique UUID identifier of the logged transaction.
    account_id : str
        The account identifier.
    before_time : float
        The timestamp of the investigated transaction.

    Returns:
    --------
    dict
        A structured JSON response with keys: decision, confidence, rationale, tools_used.
    """
    # 1. Read API Key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # Fallback: check for .env file in the current working directory
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, 'r') as ef:
                for line in ef:
                    line_stripped = line.strip()
                    if line_stripped and not line_stripped.startswith("#"):
                        parts = line_stripped.split("=", 1)
                        if len(parts) == 2 and parts[0].strip() == "GROQ_API_KEY":
                            api_key = parts[1].strip().strip("'").strip('"')
                            os.environ["GROQ_API_KEY"] = api_key
                            break
                            
    if not api_key:
        return {
            "error": "GROQ_API_KEY environment variable is not configured. Unable to run ReAct loop.",
            "decision": "escalate",
            "confidence": 0.5,
            "rationale": "Investigation failed due to missing Groq API configurations. Make sure to define GROQ_API_KEY in your environment or a .env file.",
            "tools_used": []
        }

    # 2. Retrieve target transaction details directly from predictions_log.jsonl
    try:
        log_path = "predictions_log.jsonl"
        target_tx = None
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("transaction_id") == transaction_id:
                                target_tx = data
                                break
                        except Exception:
                            pass
                            
        if not target_tx:
            return {
                "error": f"Transaction ID {transaction_id} not found in the prediction logs.",
                "decision": "escalate",
                "confidence": 0.5,
                "rationale": "Target transaction details could not be found in the system log file.",
                "tools_used": []
            }
            
        amount = float(target_tx['amount'])
        
        # Mock identity category assignment
        from investigation_context import get_mock_identity_for_id
        _, merchant_category = get_mock_identity_for_id(transaction_id)
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve target transaction data: {str(e)}",
            "decision": "escalate",
            "confidence": 0.5,
            "rationale": "Failed to load transaction data records from database.",
            "tools_used": []
        }

    # 3. System Prompt
    system_prompt = (
        "You are an expert fraud investigation analyst. Your goal is to evaluate the risk of a target transaction "
        "and choose one of three decisions:\n"
        "  - 'block': Select this if there is strong, clear evidence of fraud (e.g. extremely high model score, "
        "significant deviation in historical amount or location, or velocity spikes).\n"
        "  - 'escalate': Select this if there is suspicious activity but it is ambiguous and requires human review.\n"
        "  - 'dismiss': Select this if the transaction matches the user's historical patterns and has low model score.\n\n"
        "You have access to context-gathering tools. Call them as needed to gather information. "
        "Do not hardcode or use a fixed sequence of tool calls; make decisions dynamically based on previous findings.\n\n"
        "At the end of your analysis, you MUST provide a final answer formatted as a JSON object matching this schema:\n"
        "{\n"
        '  "decision": "block" | "escalate" | "dismiss",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "rationale": "plain-English explanation citing specific evidence gathered",\n'
        '  "tools_used": ["list", "of", "tool", "names", "called"]\n'
        "}\n"
        "Make sure to output ONLY the JSON object for your final answer without any markdown wrap or extra commentary."
    )

    # 4. Starting Context
    start_context = (
        f"Target Transaction for Investigation:\n"
        f"  - Transaction ID: {transaction_id}\n"
        f"  - Account ID: {account_id}\n"
        f"  - Time: {before_time} seconds\n"
        f"  - Amount: ${amount:.2f}\n"
        f"  - Merchant Category: {merchant_category}\n\n"
        f"Determine the risk level. Start by calling relevant tools to review account history, pattern deviations, and model score."
    )

    # 5. Initialize Groq Client and convert tools
    client = Groq(api_key=api_key)
    openai_tools = convert_anthropic_to_openai_tools(TOOL_DEFINITIONS)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": start_context}
    ]
    
    tools_used = []

    # 6. ReAct Loop (Max 6 Iterations)
    max_iterations = 6
    for iteration in range(max_iterations):
        print(f"\n--- ReAct Loop Cycle {iteration + 1}/{max_iterations} ---")
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.0
            )
        except RateLimitError:
            return {
                "error": "rate_limit",
                "message": "Investigation temporarily unavailable (API quota reached). Please try again in a moment."
            }
        except APIStatusError as ase:
            if ase.status_code == 429:
                return {
                    "error": "rate_limit",
                    "message": "Investigation temporarily unavailable (API quota reached). Please try again in a moment."
                }
            return {
                "error": f"Groq API error (Status {ase.status_code}): {ase.message}",
                "decision": "escalate",
                "confidence": 0.5,
                "rationale": f"API transaction request failed (Status {ase.status_code}): {ase.message}",
                "tools_used": tools_used
            }
        except Exception as e:
            return {
                "error": f"Groq API call failed: {str(e)}",
                "decision": "escalate",
                "confidence": 0.5,
                "rationale": f"API transaction request failed: {str(e)}",
                "tools_used": tools_used
            }

        choice = response.choices[0]
        message = choice.message
        
        # Process and log thoughts if any content is generated
        if message.content:
            print(f"[Thoughts]\n{message.content.strip()}\n")
            
        tool_calls = message.tool_calls
        if not tool_calls:
            # End loop if final answer (JSON decision block) is returned
            messages.append(message)
            print("No tool calls requested. Ending ReAct loop.")
            break
            
        # Append assistant message requesting tool calls to history
        messages.append(message)
        
        # Execute each requested tool call
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"[Tool Request] {tool_name} with arguments: {tool_args}")
            tools_used.append(tool_name)
            
            try:
                if tool_name == "tool_get_account_history":
                    acc_id = str(tool_args.get("account_id"))
                    b_time = float(tool_args.get("before_time"))
                    result = tool_get_account_history(account_id=acc_id, before_time=b_time)

                elif tool_name == "tool_get_pattern_deviation":
                    acc_id = str(tool_args.get("account_id"))
                    tx_id = str(tool_args.get("transaction_id"))
                    b_time = float(tool_args.get("before_time"))
                    result = tool_get_pattern_deviation(account_id=acc_id, transaction_id=tx_id, before_time=b_time)

                elif tool_name == "tool_get_model_score":
                    tx_id = str(tool_args.get("transaction_id"))
                    result = tool_get_model_score(transaction_id=tx_id)

                else:
                    result = {"error": f"Tool '{tool_name}' is not recognized."}
            except Exception as ex:
                result = {"error": f"Error executing tool: {str(ex)}"}

            print(f"[Tool Response] {json.dumps(result)[:150]}...")
            
            # Append tool result to the history in OpenAI/Groq tool format
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result)
            })

    # 7. Extract final decision text
    final_text = ""
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "assistant":
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if content:
                final_text = content
                break

    try:
        final_decision = _extract_json(final_text)
        
        # Standardize return keys
        final_decision["decision"] = final_decision.get("decision", "escalate").lower()
        if final_decision["decision"] not in ["block", "escalate", "dismiss"]:
            final_decision["decision"] = "escalate"

        final_decision["confidence"] = float(final_decision.get("confidence", 0.5))
        final_decision["rationale"] = str(final_decision.get("rationale", "No explanation provided."))
        
        # Sync tools used
        actual_tools_called = list(set(tools_used))
        final_decision["tools_used"] = actual_tools_called
        
        return final_decision

    except Exception as e:
        return {
            "error": f"Failed to parse structured JSON result: {str(e)}",
            "decision": "escalate",
            "confidence": 0.5,
            "rationale": f"Parsing failed for final analysis response. Raw response: {final_text}",
            "tools_used": list(set(tools_used))
        }

if __name__ == "__main__":
    # Test block execution
    # Ensure GROQ_API_KEY environment variable is verified
    api_key_check = os.environ.get("GROQ_API_KEY")
    if not api_key_check:
        raise ValueError("GROQ_API_KEY environment variable is not set. Cannot run verification tests.")
    
    # Register mock log entry for 541 into predictions_log.jsonl if not exists
    import os
    import json
    from agent_tools import _load_data
    df = _load_data()
    tx_raw = df.iloc[541]
    
    sample_id = "541-uuid-placeholder-for-test"
    from investigation_context import get_mock_identity_for_id
    account_id, _ = get_mock_identity_for_id(sample_id)
    before_time = float(tx_raw["Time"])
    
    log_entry = {
        "transaction_id": sample_id,
        "timestamp": "2026-07-03T04:00:00.000000+00:00",
        "time": before_time,
        "amount": float(tx_raw["Amount"]),
        "fraud_probability": 0.9982,
        "is_fraud": True,
        "risk_level": "high"
    }
    for i in range(1, 29):
        log_entry[f"V{i}"] = float(tx_raw[f"V{i}"])
        
    log_path = "predictions_log.jsonl"
    existing_ids = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as lf:
            for line in lf:
                if line.strip():
                    try:
                        existing_ids.append(json.loads(line).get("transaction_id"))
                    except Exception:
                        pass
    if sample_id not in existing_ids:
        with open(log_path, "a") as lf:
            lf.write(json.dumps(log_entry) + "\n")
            
    print(f"Running ReAct loop for transaction UUID {sample_id} (Account: {account_id}, Time: {before_time})...")
    decision = investigate_transaction(
        transaction_id=sample_id,
        account_id=account_id,
        before_time=before_time
    )
    print("\n==================================================")
    print("FINAL AGENT DECISION REPORT")
    print("==================================================")
    print(json.dumps(decision, indent=2))
