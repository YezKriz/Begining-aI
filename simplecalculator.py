from fastapi import FastAPI,HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse

app = FastAPI()
    title="Simple Calculator API",
    description="A simple calculator API"
    version="1.0.0",

class CalculatorRequest(BaseModel):
    num1: float
    num2: float
    operation: str


@app.post("/calculate")
def calculate(request: CalculatorRequest):
    if request.operation == "add":
        result = request.num1 + request.num2

    elif request.operation == "subtract":
        result = request.num1 - request.num2

    elif request.operation == "multiply":
        result = request.num1 * request.num2

    elif request.operation == "divide":
        if request.num2 == 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot divide by zero"
            )
        result = request.num1 / request.num2

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid operation"
        )

    return {
        "num1": request.num1,
        "num2": request.num2,
        "operation": request.operation,
        "result": result
    }