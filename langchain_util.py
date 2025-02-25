import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, api_key=anthropic_api_key)

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)


df["text"] = df.apply(lambda row: (
    f"Passenger {row['Name']} ({row['Sex']}), age {row['Age']}, traveled in class {row['Pclass']} with ticket number {row['Ticket']}. "
    f"They paid a fare of {row['Fare']} and embarked from {row['Embarked']}. "
    f"They had {row['SibSp']} siblings/spouses and {row['Parch']} parents/children aboard. "
    f"They {'survived' if row['Survived'] else 'did not survive'}."
), axis=1)

loader = DataFrameLoader(df, page_content_column="text")
documents = loader.load()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)


qa_chain = load_qa_chain(llm, chain_type="stuff")


def detect_query_type(question):
    """Detect if the query is asking for statistics (mean, count) or visualization."""
    stats_keywords = ["mean", "average", "median", "sum", "count", "min", "max", "std", "variance"]
    visual_keywords = ["distribution", "histogram", "bar chart", "scatter", "correlation", "boxplot", "visualize"]

    question_lower = question.lower()

   
    if any(keyword in question_lower for keyword in stats_keywords):
        return "statistics"

    
    if any(keyword in question_lower for keyword in visual_keywords):
        return "visual"

    
    return "text"


def handle_statistics_query(question, df):
    """Extract statistics like mean, count, min, max from dataset if asked."""
    if "fare" in question.lower() and "mean" in question.lower():
        return f"The mean fare of passengers is **${df['Fare'].mean():.2f}**."
    if "fare" in question.lower() and "median" in question.lower():
        return f"The median fare of passengers is **${df['Fare'].median():.2f}**."
    if "age" in question.lower() and "mean" in question.lower():
        return f"The mean age of passengers is **{df['Age'].mean():.2f}** years."
    if "passenger count" in question.lower():
        return f"The total number of passengers is **{df.shape[0]}**."

    return None 


def visualize_data(question, df):
    """Dynamically generates visualization based on user query."""
    if "distribution of fare" in question.lower():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Fare"].dropna(), bins=20, kde=True)
        plt.title("Fare Distribution")
        return fig

    if "bar chart of embarked passengers" in question.lower():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Embarked")
        plt.title("Passenger Embarkation Count")
        return fig

    return None 


def query_titanic(question):
    """Handles text-based responses using Claude LLM."""
    docs = vectorstore.similarity_search(question, k=100)
    try:
        response = qa_chain.run(input_documents=docs, question=question)
    except Exception:
        response = "An error occurred while processing your query."
    return response


def generate_response(question):
    """Decides whether to return statistics, visualization, or text response."""
    query_type = detect_query_type(question)

    if query_type == "statistics":
        stats_response = handle_statistics_query(question, df)
        if stats_response:
            return "text", stats_response

    if query_type == "visual":
        visualization = visualize_data(question, df)
        if visualization:
            return "visual", visualization 

    
    response_text = query_titanic(question)
    return "text", response_text
