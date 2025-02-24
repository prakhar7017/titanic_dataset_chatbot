from fastapi import FastAPI
import pandas as pd
from langchain_util import query_titanic

app = FastAPI()

# Load the Titanic dataset
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Chatbot API!"}

@app.get("/passenger_count")
def passenger_count():
    """Returns the total number of passengers."""
    count = len(df)
    return {"total_passengers": count}

@app.get("/male_percentage")
def male_percentage():
    """Returns the percentage of male passengers."""
    male_count = len(df[df["Sex"] == "male"])
    percentage = (male_count / len(df)) * 100
    return {"male_percentage": round(percentage, 2)}

@app.get("/average_fare")
def average_fare():
    """Returns the average ticket fare."""
    avg_fare = df["Fare"].mean()
    return {"average_fare": round(avg_fare, 2)}

@app.get("/query")
def ask_question(q: str):
    """Allows users to ask questions in natural language."""
    response = query_titanic(q)
    return {"question": q, "answer": response}
