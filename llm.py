# ============================================================
# FILE: app/services/llm.py
# PURPOSE: Async OpenAI chat completion with RAG prompt template
# ============================================================

from openai import AsyncOpenAI, APIError

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions based ONLY on the provided context. "
    "If the answer is not in the context, say 'I don't know based on the provided document.'"
)

_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


async def generate_answer(context_chunks: list[str], question: str) -> str:
    """Call gpt-4o-mini with the retrieved context and return the answer string.

    Args:
        context_chunks: List of relevant text passages retrieved from FAISS.
        question:       The user's question.

    Returns:
        The model's answer as a plain string.

    Raises:
        APIError: On unrecoverable OpenAI API failures.
    """
    context = "\n\n---\n\n".join(context_chunks)
    user_message = _USER_TEMPLATE.format(context=context, question=question)

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
    except APIError as exc:
        logger.error("OpenAI chat completion failed: %s", exc)
        raise

    usage = response.usage
    if usage:
        logger.info(
            "Token usage — prompt: %d, completion: %d, total: %d",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

    answer = response.choices[0].message.content or ""
    logger.info("Generated answer (%d chars)", len(answer))
    return answer.strip()
