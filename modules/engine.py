from typing import List, TypedDict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph


class Config:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHROMA_DIR = "./chroma_db"
    LLM_MODEL = "qwen2.5:3b"
    CHUNK_K = 3

class HELIOSState(TypedDict, total=False):
    question: str
    generation: str
    documents: List[any]
    is_relevant: bool
    sources: str


class HELIOSEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model=Config.EMBEDDING_MODEL)
        self.db = Chroma(embedding_function=self.embeddings, persist_directory=Config.CHROMA_DIR)
        self.llm = OllamaLLM(model=Config.LLM_MODEL, temperature=0)


    def retrieve(self, state: HELIOSState) -> HELIOSState:
        """Retrieves relevant documents from the Chroma database based on the question in the state.
        """
        print(f"---RETRIEVING DOCUMENTS FOR QUESTION: {state['question']}---")
        question = state["question"]
        documents = self.db.similarity_search(question, k=Config.CHUNK_K)
        return {
            "documents": documents
        }

    def grade_relevance(self, state: HELIOSState) -> HELIOSState:
        """Grades the relevance of the retrieved documents to the question.

        Added debug printing to show what the state contains when this node runs
        so we can see whether ``sources`` is arriving correctly.
        """
        print(f"---GRADING RELEVANCE OF DOCUMENTS---")
        print("state at grade_relevance: ", state)
        question = state.get("question")
        documents = state.get("documents")
        if not documents:
            return {"is_relevant": False}
        # simplistic check, can be replaced with actual scoring logic
        return {"is_relevant": True}

    def generate_answer(self, state: HELIOSState) -> HELIOSState:
        """Generates an answer to the question based on the retrieved documents.

        We also propagate formatted sources (if available) so the final output
        includes them despite how the graph merges node outputs.
        """
        print(f"---GENERATING ANSWER---")
        question = state.get("question")
        documents = state.get("documents", [])

        generation_prompt = f"Question: {question}\n\nDocuments:\n"
        for i, doc in enumerate(documents):
            generation_prompt += f"{i+1}. {doc.page_content}\n"
        generation_prompt += "\nGenerate a concise answer to the question based on the above documents."

        # OllamaLLM is not callable; use invoke() to generate text
        generation_response = self.llm.invoke(generation_prompt)
        result = {"generation": generation_response}
        # copy sources through explicitly
        if "sources" in state:
            result["sources"] = state["sources"]
        return result


    def format_sources(self, state: HELIOSState) -> HELIOSState:
        """Formats source information from the documents in the state.
        Returns a new state field ``sources`` containing the formatted string.

        Added logs so we can verify that this node actually executes and what it produces.
        """
        print("---FORMATTING SOURCES---")
        documents = state.get("documents", [])
        if not documents:
            print("No documents found, skipping source formatting.")
            return {}

        # Extract unique source names and pages
        seen = set()
        clean_sources = []

        for doc in documents:
            name = doc.metadata.get('source', 'Unknown Doc').split('\\')[-1]  # filename only
            page = doc.metadata.get('page', 'N/A')
            entry = f"{name} (Page {page})"

            if entry not in seen:
                clean_sources.append(f"• {entry}")
                seen.add(entry)

        formatted = "\n\n   Verified Sources:  \n" + "\n".join(clean_sources)
        print("Formatted sources:", formatted)
        return {"sources": formatted}

def build_graph():
    engine = HELIOSEngine()
    workflow = StateGraph(HELIOSState)

    # use add_sequence for a straightforward linear pipeline
    workflow.add_sequence([
        ("retrieve_documents", engine.retrieve),
        ("format_sources", engine.format_sources),
        ("grade_relevance", engine.grade_relevance),
    ])
    # register generation separately since it's conditional
    workflow.add_node("generate_answer", engine.generate_answer)

    workflow.set_entry_point("retrieve_documents")

    # continue to answer only when relevance check passes
    workflow.add_conditional_edges(
        "grade_relevance",
        lambda state: state.get("is_relevant", False),
        {True: "generate_answer", False: END},
    )

    return workflow.compile()

helios_app = build_graph()

