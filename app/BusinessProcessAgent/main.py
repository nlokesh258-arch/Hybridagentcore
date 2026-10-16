
import os
import uuid
from datetime import datetime, timezone

from strands import Agent, tool
from strands.models import BedrockModel
from bedrock_agentcore.runtime import BedrockAgentCoreApp


app = BedrockAgentCoreApp()

MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "apac.amazon.nova-lite-v1:0"
)


@tool
def calculate_total(quantity: int, unit_price: float) -> dict:
    """Calculate the total price for a quantity of items."""

    if quantity <= 0:
        return {"error": "Quantity must be greater than zero."}

    if unit_price < 0:
        return {"error": "Unit price cannot be negative."}

    return {
        "quantity": quantity,
        "unit_price": unit_price,
        "total": round(quantity * unit_price, 2)
    }


@tool
def classify_business_request(description: str) -> dict:
    """Classify a customer request and recommend the next action."""

    text = description.lower()

    if any(word in text for word in
           ["invoice", "payment", "refund", "billing"]):
        category = "Billing"
        priority = "High"
        action = "Review billing details and verify the transaction."

    elif any(word in text for word in
             ["login", "password", "sign in", "account access"]):
        category = "Account Access"
        priority = "High"
        action = "Investigate the account access issue."

    elif any(word in text for word in
             ["error", "bug", "crash", "technical", "software"]):
        category = "Technical Support"
        priority = "Medium"
        action = "Collect error details and investigate the issue."

    else:
        category = "General Request"
        priority = "Medium"
        action = "Review the request and determine the right team."

    return {
        "category": category,
        "suggested_priority": priority,
        "recommended_action": action
    }


@tool
def create_business_task(
    title: str,
    priority: str,
    owner: str
) -> dict:
    """Create a demonstration task without external storage."""

    if not title.strip():
        return {"error": "Task title cannot be empty."}

    if not owner.strip():
        return {"error": "Task owner cannot be empty."}

    if priority.lower() not in {"low", "medium", "high"}:
        return {"error": "Priority must be Low, Medium, or High."}

    return {
        "task_id": "DEMO-" + uuid.uuid4().hex[:8].upper(),
        "title": title,
        "priority": priority.title(),
        "owner": owner,
        "status": "Draft created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage": "Demo only; not saved to a database."
    }


model = BedrockModel(model_id=MODEL_ID)

agent = Agent(
    model=model,
    tools=[
        calculate_total,
        classify_business_request,
        create_business_task
    ],
    system_prompt="""
You are an AI Business Process Assistant.

Understand the user's request and choose the appropriate tool.

Rules:
- Use calculate_total for item-cost calculations.
- Use classify_business_request to classify customer issues.
- Use create_business_task when asked to prepare a business task.
- Explain tool results clearly.
- Never claim a draft task is saved in a real system.
- Ask a clarifying question if essential information is missing.
- Do not claim to have performed actions that did not happen.
"""
)


@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "")

    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Please provide a non-empty prompt."}

    result = agent(prompt)
    return {"result": str(result)}


if __name__ == "__main__":
    app.run()
