from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama


load_dotenv()

model = ChatOllama(model="llama3:latest")


class Movie(BaseModel):

    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: Optional[List[str]] = None
    plot: str
    setting: Optional[str] = None
    themes: Optional[List[str]] = None
    ratings: Optional[dict] = None
    notable_features: Optional[List[str]] = None
    summary: str


parser = PydanticOutputParser(
    pydantic_object=Movie
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a movie information extraction assistant.

Extract information from the paragraph and return ONLY valid JSON.

IMPORTANT:
- Return ONLY the JSON object.
- Do NOT write anything before or after the JSON.
- Do NOT use markdown.
- Do NOT use ```json.
- Do NOT add comments.
- Use double quotes for all keys and string values.
- If information is unavailable, use null.

{format_instructions}
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])


# Streamlit UI
st.title("Movie Information Extractor")

paragraph = st.text_area("Give your paragraph:")

if st.button("Extract Movie Information"):

    final_prompt = prompt.invoke({
        "paragraph": paragraph,
        "format_instructions": parser.get_format_instructions()
    })

    response = model.invoke(final_prompt)

    # Convert LLM output into Pydantic object
    movie = parser.parse(response.content)

    st.write(movie)