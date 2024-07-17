import os
import docx2txt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI

app = FastAPI()

class Question(BaseModel):
    question: constr(min_length=1)


def read_document(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        text = docx2txt.process(file_path)
        return text, [Document(text=text)]
    except Exception as e:
        raise ValueError(f"Error reading document: {str(e)}")

def setup_llama_index(documents) -> VectorStoreIndex:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    try:
        llm = OpenAI(api_key=api_key)
        index = VectorStoreIndex.from_documents(documents, llm=llm)
        return index
    except Exception as e:
        raise ValueError(f"Error setting up LlamaIndex: {str(e)}")

@app.post("/ask")
async def ask(question: Question):
    try:
        document_text, documents = read_document(os.getenv("DOCUMENT_PATH"))
        index = setup_llama_index(documents)

        query_engine = index.as_query_engine()
        response = query_engine.query(question.question)
        return {"answer": response.response}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
