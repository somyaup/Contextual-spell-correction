from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from POC import POC
import uvicorn

app = FastAPI()
poc_obj = POC()

class QueryInput(BaseModel):
    queries: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "queries": ["hams", "goosebrry pickkle", "titkan watch", "lokal trian", "ranbow toyss stffed"]
            }
        }

class QueryOutput(BaseModel):
    corrections: dict

@app.post("/DYM/v1/corrections")
def get_corrections(queries_input: QueryInput) -> QueryOutput:
    queries = queries_input.queries
    corrections = {}

    for query in queries:
        corrections[query] = {}
        corrections[query] = poc_obj.did_you_mean(query)

    return corrections#QueryOutput(corrections=corrections)

def __main__():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    __main__()
