"""RAG chain construction and retrieval formatting helpers."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def format_retrieved_docs(docs: list[Document]) -> str:
    """Convert retrieved Documents into a prompt-ready context block.

    We include source + page in each line to improve grounding and downstream
    citation visibility in logs.
    """
    return "\n\n".join(
        f"{doc.metadata['source']} (page {doc.metadata['page']}): {doc.page_content}"
        for doc in docs
    )


def build_rag_chain(llm: ChatOpenAI):
    """Build a simple prompt -> LLM -> text parser chain.

    Input expected at invoke time:
    - context: formatted retrieval context
    - question: user question

    Output:
    - plain answer text (string)
    """
    prompt_template = ChatPromptTemplate.from_template(
        "Use only the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "If the answer is not present, say \"I don't know.\""
    )
    return prompt_template | llm | StrOutputParser()


def citations_from_docs(docs: list[Document]) -> list[str]:
    """Return deduplicated, sorted source/page citation labels."""
    return sorted(
        {
            f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
            for doc in docs
        }
    )
