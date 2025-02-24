import os
import anthropic
from langchain_anthropic import ChatAnthropic

from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from dotenv import load_dotenv
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY. Please set it in your .env file.")

# Initialize Claude LLM
llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0,
    api_key=anthropic_api_key  # Explicitly pass the API key
)

# Use Hugging Face Embeddings (instead of OpenAI)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Titanic dataset
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)

# Convert DataFrame to LangChain format
df["text"] = df.apply(lambda row: (
    f"Passenger {row['Name']} ({row['Sex']}), age {row['Age']}, traveled in class {row['Pclass']} with ticket number {row['Ticket']}. "
    f"They paid a fare of {row['Fare']} and embarked from {row['Embarked']}. "
    f"They had {row['SibSp']} siblings/spouses and {row['Parch']} parents/children aboard. "
    f"They {'survived' if row['Survived'] else 'did not survive'}."
), axis=1)

loader = DataFrameLoader(df, page_content_column="text")
documents = loader.load()

# Convert dataset into a FAISS vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Load the QA model
qa_chain = load_qa_chain(llm, chain_type="stuff")

def query_titanic(question: str):
    print(question)
    # logging.info(f"Query received: {question}")
    #  all_docs = vectorstore.get_all_documents()  
    docs = vectorstore.similarity_search(question,k=100)
    # logging.info(f"Documents retrieved: {docs}")

    try:
        response = qa_chain.run(input_documents=docs, question=question)
        logging.info(f"Response: {response}")
    except Exception as e:
        logging.error(f"Error during LangChain processing: {e}", exc_info=True)
        response = "An error occurred while processing your query."

    return response


def visualize_embarked_data():
    st.subheader("📊 Titanic Passenger Embarkation Data")

    # Clean missing values
    df_clean = df.dropna(subset=["Embarked"])

    # Count passengers per embarkation port
    embark_counts = df_clean["Embarked"].value_counts().reset_index()
    embark_counts.columns = ["Port", "Count"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=embark_counts, x="Port", y="Count", hue="Port", palette="pastel", ax=ax)
    
    plt.xlabel("Embarkation Port")
    plt.ylabel("Number of Passengers")
    plt.title("Passengers Embarked from Each Port")

    # Display the plot in Streamlit
    st.pyplot(fig)