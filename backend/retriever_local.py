import os
from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def groq_llm(prompt: str):
    """Call Groq fast free LLM."""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # FREE + FAST
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


def answer_query(retriever, question):
    """Answer a question using context + Groq LLM."""

    # ----------------------------------------
    # Retrieve relevant documents (correct way)
    # ----------------------------------------
    docs = retriever._get_relevant_documents(question, run_manager=None)

    # Build context text
    context_text = "\n\n".join([d.page_content for d in docs])

    # ----------------------------------------
    # Professional ChatGPT-style prompt
    # ----------------------------------------
    template = """
You are an advanced knowledge-base assistant.

Rules:
1. Use the provided context when relevant.
2. If the context does NOT contain the answer, reply using general knowledge.
3. NEVER say “I don’t know”.
4. ALWAYS give a clear, helpful explanation.
5. NEVER invent citations or sources.

Context:
{context}

Question:
{question}

Final Answer:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(
        context=context_text,
        question=question
    )

    # ----------------------------------------
    # Call Groq LLM
    # ----------------------------------------
    answer_text = groq_llm(final_prompt)

    # ----------------------------------------
    # Build clean, deduped sources
    # ----------------------------------------
    unique_sources = []
    seen = set()

    for doc in docs:
        src = doc.metadata.get("source", "unknown")

        if src not in seen:
            seen.add(src)
            unique_sources.append({
                "source": src,
                "snippet": doc.page_content[:300].replace("\n", " ")
            })

    return {
        "answer": answer_text,
        "sources": unique_sources
    }









# import os
# from groq import Groq
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# def groq_llm(prompt: str):
#     """Call Groq LLM."""
#     res = client.chat.completions.create(
#         model="llama-3.1-8b-instant",   # FREE + VERY FAST
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return res.choices[0].message.content


# def answer_query(retriever, question):

#     template = """
#    You are a helpful AI assistant.
#    Use the provided context if it is relevant information.

#    If the context does NOT contain information related to the question:  
#     - answer using your general knowledge
#     - NEVER say "I don't know"
#     - give a clear and helpful explanation
#     - do NOT make up citations.


#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """

#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["context", "question"]
#     )

#     # Get relevant documents for context
#     docs = retriever.invoke(question)
#     context_text = "\n\n".join([d.page_content for d in docs])

#     # Convert prompt → final text
#     final_prompt = prompt.format(
#         context=context_text,
#         question=question
#     )

#     # Call Groq
#     answer_text = groq_llm(final_prompt)

#     # Build sources list
#     sources = [{
#         "source": doc.metadata.get("source", "unknown"),
#         "snippet": doc.page_content[:300]
#     } for doc in docs]

#     return {"answer": answer_text, "sources": sources}
