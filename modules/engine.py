import os
from typing import Any, List, TypedDict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph


class Config:
    EMBEDDING_MODEL = os.getenv("HELIOS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHROMA_DIR = os.getenv("HELIOS_CHROMA_DIR", "./chroma_db")
    LLM_MODEL = os.getenv("HELIOS_LLM_MODEL", "qwen2.5:3b")
    OLLAMA_BASE_URL = os.getenv("HELIOS_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    CHUNK_K = int(os.getenv("HELIOS_CHUNK_K", "6"))


class HELIOSState(TypedDict, total=False):
    question: str
    generation: str
    documents: List[Any]
    is_relevant: bool
    sources: str
    iteration: int
    trace: str


class HELIOSEngine:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )

        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_DIR
        )

        self.llm = OllamaLLM(
            model=Config.LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0
        )

    # -------------------------
    # RETRIEVAL
    # -------------------------

    def retrieve(self, state: HELIOSState) -> HELIOSState:

        question = state["question"]
        iteration = state.get("iteration", 1)

        print(f"--- RETRIEVAL ITERATION {iteration} ---")

        docs = self.db.max_marginal_relevance_search(
            question,
            k=Config.CHUNK_K,
            fetch_k=20
        )

        return {
            "documents": docs,
            "iteration": iteration,
            "trace": f"Iteration {iteration}: Retrieved {len(docs)} documents"
        }

    # -------------------------
    # RELEVANCE CHECK
    # -------------------------

    def grade_relevance(self, state: HELIOSState) -> HELIOSState:

        docs = state.get("documents", [])

        if not docs:
            return {
                "is_relevant": False,
                "trace": state.get("trace", "") + " → No documents found"
            }

        return {
            "is_relevant": True,
            "trace": state.get("trace", "") + " → Documents validated"
        }

    # -------------------------
    # FORMAT SOURCES
    # -------------------------

    def format_sources(self, state: HELIOSState) -> HELIOSState:

        docs = state.get("documents", [])

        if not docs:
            return {}

        sources = []

        for i, doc in enumerate(docs):

            name = doc.metadata.get("source", "Unknown").split("\\")[-1]
            page = doc.metadata.get("page", "N/A")

            sources.append(f"[{i+1}] {name} (Page {page})")

        formatted = "\n".join(sources)

        return {"sources": formatted}

    # -------------------------
    # GENERATE ANSWER
    # -------------------------

    def generate_answer(self, state: HELIOSState) -> HELIOSState:

        question = state["question"]
        docs = state.get("documents", [])

        context = ""

        for i, doc in enumerate(docs):
             context += f"""
            SOURCE [{i+1}]
            {doc.page_content}
            """

        prompt = f"""
You are a medical research assistant.

Answer the question using ONLY the provided sources.

Rules:
- Cite sources using numbers like [1], [2], [3]
- Do not invent information
- If the answer is not found say:
"The answer is not found in the retrieved documents."

Question:
{question}

Sources:
{context}

Answer with citations.
"""

        try:
            response = self.llm.invoke(prompt)
            return {"generation": response}
        except Exception as exc:
            error_message = (
                f"Answer generation is temporarily unavailable because Ollama "
                f"could not run model '{Config.LLM_MODEL}'. "
                f"Underlying error: {exc}"
            )
            return {
                "generation": error_message,
                "trace": state.get("trace", "") + " → Generation failed"
            }


# -------------------------
# BUILD GRAPH
# -------------------------

def build_graph():

    engine = HELIOSEngine()

    workflow = StateGraph(HELIOSState)

    workflow.add_node("retrieve", engine.retrieve)
    workflow.add_node("format_sources", engine.format_sources)
    workflow.add_node("grade_relevance", engine.grade_relevance)
    workflow.add_node("generate", engine.generate_answer)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "format_sources")
    workflow.add_edge("format_sources", "grade_relevance")

    workflow.add_conditional_edges(
        "grade_relevance",
        lambda state: state.get("is_relevant", False),
        {
            True: "generate",
            False: END
        }
    )

    workflow.add_edge("generate", END)

    return workflow.compile()


helios_app = build_graph()
