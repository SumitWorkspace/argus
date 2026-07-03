import os
import json
import re
import pandas as pd
from google import genai
from google.genai import types

# Import tool definitions and backend implementations
from agent_tools import (
    TOOL_DEFINITIONS,
    tool_get_account_history,
    tool_get_pattern_deviation,
    tool_get_model_score,
    _load_data
)

def convert_anthropic_to_gemini_tools(anthropic_tools):
    """
    Translates tool definitions from Anthropic format (name, description, input_schema)
    into Gemini's SDK Schema representation.
    """
    type_map = {
        "object": types.Type.OBJECT,
        "string": types.Type.STRING,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY
    }
    
    function_declarations = []
    for tool in anthropic_tools:
        name = tool["name"]
        description = tool["description"]
        input_schema = tool.get("input_schema", {})
        
        # Build properties
        properties = {}
        anthropic_props = input_schema.get("properties", {})
        for prop_name, prop_val in anthropic_props.items():
            prop_type_str = prop_val.get("type", "string").lower()
            prop_type = type_map.get(prop_type_str, types.Type.STRING)
            
            properties[prop_name] = types.Schema(
                type=prop_type,
                description=prop_val.get("description", "")
            )
            
        required = input_schema.get("required", [])
        
        parameters = types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            required=required
        )
        
        fd = types.FunctionDeclaration(
            name=name,
            description=description,
            parameters=parameters
        )
        function_declarations.append(fd)
        
    return [types.Tool(function_declarations=function_declarations)]

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

def investigate_transaction(transaction_index, account_id, before_time):
    """
    Runs a ReAct-style agent loop using the Google Gemini API to analyze a target transaction's fraud risk.
    Gathers evidence using tools and returns a structured risk decision.

    Parameters:
    -----------
    transaction_index : int
        The row index of the transaction in the dataset.
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
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {
            "error": "GOOGLE_API_KEY environment variable is not configured. Unable to run ReAct loop.",
            "decision": "escalate",
            "confidence": 0.5,
            "rationale": "Investigation failed due to missing Google API configurations.",
            "tools_used": []
        }

    # 2. Retrieve target transaction details for starting context
    try:
        df = _load_data()
        if transaction_index < 0 or transaction_index >= len(df):
            return {
                "error": f"Transaction index {transaction_index} is out of bounds.",
                "decision": "escalate",
                "confidence": 0.5,
                "rationale": "Target transaction index is out of bounds.",
                "tools_used": []
            }
        target_tx = df.iloc[transaction_index]
        amount = float(target_tx['Amount'])
        merchant_category = str(target_tx['merchant_category'])
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
        f"  - Transaction Index: {transaction_index}\n"
        f"  - Account ID: {account_id}\n"
        f"  - Time: {before_time} seconds\n"
        f"  - Amount: ${amount:.2f}\n"
        f"  - Merchant Category: {merchant_category}\n\n"
        f"Determine the risk level. Start by calling relevant tools to review account history, pattern deviations, and model score."
    )

    # 5. Initialize Gemini Client and convert tools
    client = genai.Client(api_key=api_key)
    gemini_tools = convert_anthropic_to_gemini_tools(TOOL_DEFINITIONS)
    
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=gemini_tools,
        temperature=0.0
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=start_context)]
        )
    ]
    
    tools_used = []

    # 6. ReAct Loop (Max 6 Iterations)
    max_iterations = 6
    for iteration in range(max_iterations):
        print(f"\n--- ReAct Loop Cycle {iteration + 1}/{max_iterations} ---")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config
            )
        except Exception as e:
            return {
                "error": f"Gemini API call failed: {str(e)}",
                "decision": "escalate",
                "confidence": 0.5,
                "rationale": f"API transaction request failed: {str(e)}",
                "tools_used": tools_used
            }

        # Process and log agent thoughts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    print(f"[Thoughts]\n{part.text.strip()}\n")

        # Parse and handle function calls
        tool_calls = response.function_calls
        if not tool_calls:
            # Append the final assistant response to the conversation history
            if response.candidates:
                contents.append(response.candidates[0].content)
            print("No tool calls requested. Ending ReAct loop.")
            break

        tool_responses = []
        for fc in tool_calls:
            tool_name = fc.name
            tool_args = fc.args

            print(f"[Tool Request] {tool_name} with arguments: {tool_args}")
            tools_used.append(tool_name)

            # Invoke tool dynamically
            try:
                if tool_name == "tool_get_account_history":
                    acc_id = str(tool_args.get("account_id"))
                    b_time = float(tool_args.get("before_time"))
                    result = tool_get_account_history(account_id=acc_id, before_time=b_time)

                elif tool_name == "tool_get_pattern_deviation":
                    acc_id = str(tool_args.get("account_id"))
                    tx_idx = int(tool_args.get("transaction_index"))
                    b_time = float(tool_args.get("before_time"))
                    result = tool_get_pattern_deviation(account_id=acc_id, transaction_index=tx_idx, before_time=b_time)

                elif tool_name == "tool_get_model_score":
                    tx_idx = int(tool_args.get("transaction_index"))
                    result = tool_get_model_score(transaction_index=tx_idx)

                else:
                    result = {"error": f"Tool '{tool_name}' is not recognized."}
            except Exception as ex:
                result = {"error": f"Error executing tool: {str(ex)}"}

            print(f"[Tool Response] {json.dumps(result)[:150]}...")

            # Gemini requires response payload to be a JSON object/dict
            if isinstance(result, dict):
                response_payload = result
            elif isinstance(result, list):
                response_payload = {"transactions": result}
            else:
                response_payload = {"result": result}

            # Create FunctionResponse part
            part = types.Part.from_function_response(
                name=tool_name,
                response=response_payload
            )
            tool_responses.append(part)

        # To keep conversation history correct:
        # A. Append model's response (containing tool call requests) to contents
        if response.candidates:
            contents.append(response.candidates[0].content)
        
        # B. Append the tool execution results back to model with the role 'tool'
        contents.append(
            types.Content(
                role="tool",
                parts=tool_responses
            )
        )

    # 7. Parse final decision response
    final_text = ""
    # Look at the last assistant message in the history
    last_assistant_msg = next((msg for msg in reversed(contents) if msg.role == "model"), None)
    if last_assistant_msg and last_assistant_msg.parts:
        for part in last_assistant_msg.parts:
            if part.text:
                final_text += part.text

    try:
        final_decision = _extract_json(final_text)
        
        # Standardize return keys
        final_decision["decision"] = final_decision.get("decision", "escalate").lower()
        if final_decision["decision"] not in ["block", "escalate", "dismiss"]:
            final_decision["decision"] = "escalate"

        final_decision["confidence"] = float(final_decision.get("confidence", 0.5))
        final_decision["rationale"] = str(final_decision.get("rationale", "No explanation provided."))
        
        # Deduplicate and sync tools used
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
    # Ensure GOOGLE_API_KEY environment variable is verified
    api_key_check = os.environ.get("GOOGLE_API_KEY")
    if not api_key_check:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Cannot run verification tests.")
    
    print("Running ReAct loop for transaction index 541 (Account: ACC_039, Time: 406.0)...")
    # Target transaction: index 541, account ID: ACC_039, time: 406.0 seconds
    decision = investigate_transaction(
        transaction_index=541,
        account_id="ACC_039",
        before_time=406.0
    )
    print("\n==================================================")
    print("FINAL AGENT DECISION REPORT")
    print("==================================================")
    print(json.dumps(decision, indent=2))
