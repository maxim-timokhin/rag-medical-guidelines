import os
import asyncio
import json
from dotenv import load_dotenv

# Load env before anything else
load_dotenv()

from app.agent import app as adk_app
from google.adk.runners import InMemoryRunner
from google.genai import types
from tests.eval.response_quality import evaluate

async def run_case(runner, prompt_text: str):
    session = await runner.session_service.create_session(
        app_name="app", user_id="eval_user"
    )
    
    response_text = ""
    turns = []
    
    turns.append({
        "role": "user",
        "parts": [{"text": prompt_text}]
    })
    
    async for event in runner.run_async(
        user_id="eval_user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)]),
    ):
        if event.content and event.content.parts:
            response_text += "".join(p.text for p in event.content.parts if p.text)
            
    turns.append({
        "role": "model",
        "parts": [{"text": response_text}]
    })
    
    return response_text, {"turns": turns}

async def main():
    with open("tests/eval/datasets/basic-dataset.json", "r") as f:
        dataset = json.load(f)
        
    runner = InMemoryRunner(app=adk_app)
    
    print("-" * 80)
    print("RUNNING LOCAL EVALUATION")
    print("-" * 80)
    
    results = []
    for case in dataset["eval_cases"]:
        case_id = case["eval_case_id"]
        prompt_text = case["prompt"]["parts"][0]["text"]
        
        print(f"Running case: {case_id}...")
        response, agent_data = await run_case(runner, prompt_text)
        
        instance = {
            "prompt": prompt_text,
            "response": response,
            "agent_data": agent_data,
            "reference": case.get("reference", {}).get("response", {}).get("parts", [{}])[0].get("text", "")
        }
        
        grade_res = evaluate(instance)
        results.append({
            "case_id": case_id,
            "prompt": prompt_text,
            "response": response,
            "score": grade_res["score"],
            "explanation": grade_res["explanation"]
        })
        
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    for res in results:
        print(f"Case: {res['case_id']}")
        print(f"Prompt: {res['prompt']}")
        print(f"Response: {res['response']}")
        print(f"Score: {res['score']}/5")
        print(f"Explanation: {res['explanation']}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())
