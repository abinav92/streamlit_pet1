from openai import OpenAI
import streamlit as st
from chromadb import Client
# from chromadb.utils import embedding
from langchain_openai import OpenAIEmbeddings
import os
from langchain_chroma import Chroma

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# Initialize Streamlit and OpenAI client
st.title("Chatbot with Vector Store")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize ChromaDB client and collection
chroma_client = Client()
# collection = chroma_client.get_collection("pets", embedding_function = embeddings)
collection = Chroma(
    collection_name="pets",
    embedding_function=embeddings,
    persist_directory="./chroma_pets_db",  # Where to save data locally, remove if not necessary
)

# Set model and session state for message tracking
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Retrieve and process user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Convert user input to embedding
    prompt_embedding = embeddings(prompt)

    # Step 2: Retrieve relevant documents from ChromaDB
    results = collection.query(prompt_embedding, n_results=5)  # Retrieve top 5 matches
    context_documents = [result["document"] for result in results]

    # Step 3: Add context documents to message history for OpenAI
    context_content = "\n\n".join(context_documents)  # Combine retrieved docs as context
    st.session_state.messages.append({"role": "system", "content": f"Context: {context_content}"})

    # Step 4: Generate response with OpenAI API, including context
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        
        # Collect streamed response
        response = ""
        for chunk in stream:
            response += chunk["choices"][0]["delta"].get("content", "")
            st.write_stream(chunk)

    # Save assistant's response in session state
    st.session_state.messages.append({"role": "assistant", "content": response})
