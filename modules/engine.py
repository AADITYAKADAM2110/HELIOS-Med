from typing import List, TypedDict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph


class GraphState(TypedDict, total=False):
    question: str
    generation: str
    documents: List
    iteration: int


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="qwen2.5:3b", temperature=0)


def retrieve_documents(state: GraphState) -> GraphState:
    print(f"---RETRIEVING DOCUMENTS FOR QUESTION: {state['question']}---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "iteration": state.get("iteration", 0) + 1,
    }


def generate_answer(state: GraphState) -> GraphState:
    print(f"---GENERATING ANSWER FOR QUESTION: {state['question']}---")
    question = state["question"]
    documents = state.get("documents", [])
    document_texts = "\n\n".join([doc.page_content for doc in documents])
    prompt = (
        f"Question: {question}\n\nDocuments:\n{document_texts}\n\n"
        "Based on the provided documents, generate a comprehensive answer to the question."
    )
    generation = llm.invoke(prompt)
    return {"generation": generation}


workflow = StateGraph(GraphState)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_answer")
workflow.add_edge("generate_answer", END)

helios_engine = workflow.compile()


if __name__ == "__main__":
    result = helios_engine.invoke({"question": "Does this document talk about Planes?"})
    print(result)