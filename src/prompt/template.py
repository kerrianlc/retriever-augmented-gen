TEMPLATE = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
    },
]