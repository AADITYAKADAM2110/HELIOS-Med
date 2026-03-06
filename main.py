from modules.engine import helios_app
import chainlit as cl
import json
from datetime import datetime


@cl.on_chat_start
async def start():

    welcome_message = """
# HELIOS-Med 🩺

Welcome to **HELIOS-Med**, an Agentic Retrieval-Augmented Generation system designed to assist with healthcare policy document analysis.

This system retrieves relevant information from policy documents and generates **document-grounded answers with verified citations**.

### Example Questions You Can Ask

• What is the purpose of the policy document?  
• What steps are involved in policy development?  
• How are healthcare policies approved?  
• How is policy compliance monitored?  
• What are future trends in healthcare policy governance?

Ask a question to begin.
"""

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):

    # Create orchestration step
    step = cl.Step(name="HELIOS Orchestrator")
    await step.send()

    step.input = message.content

    # Run HELIOS engine
    result = await cl.make_async(helios_app.invoke)(
        {"question": message.content}
    )

    generation = result.get("generation", "No answer generated.")
    docs = result.get("documents", [])
    sources = result.get("sources", "")
    trace = result.get("trace", "")
    iteration = result.get("iteration", 1)
    is_relevant = result.get("is_relevant", True)

    # Confidence score (simple heuristic)
    confidence = min(1.0, len(docs) / 6)
    confidence_score = f"{round(confidence * 100)}%"

    step.output = f"""
Documents Retrieved: {len(docs)}
Iteration: {iteration}
Relevance Check: {"Passed" if is_relevant else "Failed"}
Confidence Score: {confidence_score}
"""
    await step.update()

    # Create side panel document viewer
    elements = []
    seen = set()

    for doc in docs:

        name = doc.metadata.get("source", "Unknown").split("\\")[-1]
        page = doc.metadata.get("page", "N/A")

        label = f"{name} (Page {page})"

        if label not in seen:

            elements.append(
                cl.Text(
                    name=label,
                    content=doc.page_content,
                    display="side"
                )
            )

            seen.add(label)

    # Guard: no documents retrieved
    if not docs:
        await cl.Message(
            content="⚠️ No relevant documents found in the knowledge base."
        ).send()
        return

    # Guard: documents not relevant
    if not is_relevant:
        final_response = "⚠️ Retrieved documents were not relevant. Answer withheld."

    else:

        final_response = f"""
### HELIOS Answer

{generation}

---

### Confidence Score

{confidence_score}

---

### Reasoning Trace

{trace}

---

### Verified Sources

{sources}
"""

    # Send response to UI
    await cl.Message(
        content=final_response,
        elements=elements
    ).send()

    # -------------------------
    # Logging for evaluation
    # -------------------------

    docs_used = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A")
        }
        for doc in docs
    ]

    log_data = {
        "timestamp": str(datetime.now()),
        "question": message.content,
        "answer": generation,
        "sources": sources,
        "documents_used": docs_used,
        "num_documents": len(docs),
        "confidence": confidence_score
    }

    with open("ui_results.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")