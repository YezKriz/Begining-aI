# ============================================================
# LOCAL RAG APPLICATION
# LANGCHAIN + FAISS + HUGGINGFACE EMBEDDINGS + OPENAI
#
# PDF:
# AI_Empowered_SAFe_POPM_Course_Content(2).pdf
#
# PURPOSE:
# This application:
#
# 1. Loads the PDF
# 2. Splits the PDF into chunks
# 3. Creates embeddings locally
# 4. Stores embeddings in a local FAISS database
# 5. Retrieves relevant chunks for a question
# 6. Sends retrieved context to an LLM
# 7. Generates an answer
# 8. Displays the PDF pages used as sources
#
# Every major task provides:
# SUCCESS notification
# ERROR notification
# PROGRESS notification
# TIME TAKEN
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================
# PURPOSE:
# Import all libraries required by the RAG application.
# ============================================================

import os
import time

from dotenv import load_dotenv

# Location of this Python script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# .env is one folder above the Python script
ENV_PATH = os.path.join(BASE_DIR, "..", ".env")

# Load environment variables
load_dotenv(ENV_PATH)

# Read OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_openai import ChatOpenAI


# ============================================================
# 2. CONFIGURATION
# ============================================================
# PURPOSE:
# Store application settings in one place.
#
# PDF_PATH:
# Location of the PDF.
#
# FAISS_PATH:
# Local folder where FAISS database will be stored.
#
# EMBEDDING_MODEL:
# HuggingFace embedding model.
# This runs locally on your computer.
# ============================================================

# Get the folder where this Python file is located
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# PDF is in the same folder as this Python file
PDF_PATH = os.path.join(
    BASE_DIR,
    "AI_Empowered_SAFe_POPM_Course_Content.pdf"
)

# FAISS database will also be created in this folder
FAISS_PATH = os.path.join(
    BASE_DIR,
    "faiss_safe_popm"
)

# Local embedding model
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. NOTIFICATION FUNCTIONS
# ============================================================
# PURPOSE:
# These functions provide clear notifications whenever a task
# starts, completes successfully, or fails.
#
# This makes it easy to understand what the RAG application
# is doing internally.
# ============================================================


def task_started(task_name):

    print()
    print("=" * 70)
    print(f"STARTED : {task_name}")
    print("=" * 70)


def task_success(task_name, details=""):

    print()
    print("-" * 70)
    print(f"SUCCESS : {task_name}")

    if details:
        print(f"DETAILS : {details}")

    print("-" * 70)


def task_error(task_name, error):

    print()
    print("!" * 70)
    print(f"ERROR   : {task_name}")
    print(f"DETAILS : {error}")
    print("!" * 70)


def task_info(message):

    print(f"[INFO] {message}")


# ============================================================
# 4. LOAD PDF
# ============================================================
# PURPOSE:
# Read the PDF and convert it into LangChain Documents.
#
# Each PDF page becomes a Document containing:
#
# - page content
# - page metadata
# ============================================================


def load_pdf():

    task_started("PDF Loading")

    start_time = time.time()

    try:

        # Check whether PDF exists
        if not os.path.exists(PDF_PATH):

            raise FileNotFoundError(
                f"PDF not found: {PDF_PATH}"
            )

        task_info(
            f"PDF found: {PDF_PATH}"
        )

        # Create PDF loader
        loader = PyPDFLoader(PDF_PATH)

        task_info(
            "PyPDFLoader created successfully."
        )

        # Load PDF
        documents = loader.load()

        elapsed_time = time.time() - start_time

        task_success(
            "PDF Loading",
            f"{len(documents)} pages loaded "
            f"in {elapsed_time:.2f} seconds."
        )

        return documents

    except Exception as e:

        task_error(
            "PDF Loading",
            e
        )

        raise


# ============================================================
# 5. SPLIT DOCUMENT INTO CHUNKS
# ============================================================
# PURPOSE:
# Split the PDF text into smaller pieces.
#
# Why?
#
# Vector databases work better when documents are divided
# into smaller meaningful pieces.
#
# chunk_size = 800
#
# chunk_overlap = 150
#
# The overlap helps preserve context between chunks.
# ============================================================


def split_documents(documents):

    task_started("Document Chunking")

    start_time = time.time()

    try:

        task_info(
            "Creating RecursiveCharacterTextSplitter..."
        )

        text_splitter = RecursiveCharacterTextSplitter(

            chunk_size=800,

            chunk_overlap=150
        )

        task_info(
            "Splitting documents into chunks..."
        )

        chunks = text_splitter.split_documents(
            documents
        )

        elapsed_time = time.time() - start_time

        task_success(
            "Document Chunking",
            f"{len(chunks)} chunks created "
            f"in {elapsed_time:.2f} seconds."
        )

        return chunks

    except Exception as e:

        task_error(
            "Document Chunking",
            e
        )

        raise


# ============================================================
# 6. CREATE EMBEDDING MODEL
# ============================================================
# PURPOSE:
# Convert text into numerical vectors.
#
# Example:
#
# "What is PI Planning?"
#
# becomes a numerical vector.
#
# Similar meaning -> similar vectors.
#
# HuggingFace model runs locally.
# ============================================================


def create_embeddings():

    task_started("Embedding Model Initialization")

    start_time = time.time()

    try:

        task_info(
            f"Loading model: {EMBEDDING_MODEL}"
        )

        embeddings = HuggingFaceEmbeddings(

            model_name=EMBEDDING_MODEL
        )

        elapsed_time = time.time() - start_time

        task_success(
            "Embedding Model Initialization",
            f"Model loaded in "
            f"{elapsed_time:.2f} seconds."
        )

        return embeddings

    except Exception as e:

        task_error(
            "Embedding Model Initialization",
            e
        )

        raise


# ============================================================
# 7. CREATE FAISS VECTOR DATABASE
# ============================================================
# PURPOSE:
# Convert every document chunk into an embedding and store
# those embeddings inside FAISS.
#
# FAISS is stored locally.
# ============================================================


def create_vector_database(
    chunks,
    embeddings
):

    task_started(
        "FAISS Vector Database Creation"
    )

    start_time = time.time()

    try:

        task_info(
            f"Creating vectors for {len(chunks)} chunks..."
        )

        vector_db = FAISS.from_documents(

            documents=chunks,

            embedding=embeddings
        )

        task_success(
            "FAISS Vector Database Creation",
            "Embeddings successfully created "
            "and stored in FAISS."
        )

        # ----------------------------------------------------
        # SAVE FAISS DATABASE
        # ----------------------------------------------------

        task_info(
            f"Saving FAISS database to: {FAISS_PATH}"
        )

        vector_db.save_local(
            FAISS_PATH
        )

        elapsed_time = time.time() - start_time

        task_success(
            "FAISS Database Save",
            f"Database saved locally in "
            f"{elapsed_time:.2f} seconds."
        )

        return vector_db

    except Exception as e:

        task_error(
            "FAISS Vector Database Creation",
            e
        )

        raise


# ============================================================
# 8. LOAD EXISTING FAISS DATABASE
# ============================================================
# PURPOSE:
# If FAISS already exists, load it instead of processing
# the PDF again.
#
# This makes subsequent application starts much faster.
# ============================================================


def load_vector_database(
    embeddings
):

    task_started(
        "Loading Existing FAISS Database"
    )

    start_time = time.time()

    try:

        task_info(
            f"Loading FAISS database from: {FAISS_PATH}"
        )

        vector_db = FAISS.load_local(

            FAISS_PATH,

            embeddings,

            allow_dangerous_deserialization=True
        )

        elapsed_time = time.time() - start_time

        task_success(
            "Loading Existing FAISS Database",
            f"Database loaded in "
            f"{elapsed_time:.2f} seconds."
        )

        return vector_db

    except Exception as e:

        task_error(
            "Loading Existing FAISS Database",
            e
        )

        raise


# ============================================================
# 9. CREATE OR LOAD VECTOR DATABASE
# ============================================================
# PURPOSE:
#
# FIRST RUN:
#
# PDF
# ↓
# Chunks
# ↓
# Embeddings
# ↓
# FAISS
#
#
# NEXT RUN:
#
# Existing FAISS
# ↓
# Load FAISS
# ============================================================


def get_vector_database():

    task_started(
        "Vector Database Preparation"
    )

    try:

        # ----------------------------------------------
        # Load embedding model
        # ----------------------------------------------

        embeddings = create_embeddings()

        # ----------------------------------------------
        # Check whether FAISS already exists
        # ----------------------------------------------

        if os.path.exists(FAISS_PATH):

            task_info(
                "Existing FAISS database detected."
            )

            vector_db = load_vector_database(
                embeddings
            )

        else:

            task_info(
                "FAISS database does not exist."
            )

            task_info(
                "Starting first-time PDF processing..."
            )

            # Load PDF
            documents = load_pdf()

            # Split PDF
            chunks = split_documents(
                documents
            )

            # Create FAISS
            vector_db = create_vector_database(

                chunks,

                embeddings
            )

        task_success(
            "Vector Database Preparation",
            "FAISS vector database is ready."
        )

        return vector_db

    except Exception as e:

        task_error(
            "Vector Database Preparation",
            e
        )

        raise


# ============================================================
# 10. CREATE LLM
# ============================================================
# PURPOSE:
# FAISS retrieves relevant information.
#
# The LLM uses that information to generate the final answer.
#
# This example uses OpenAI.
#
# Set API key in PowerShell:
#
# $env:OPENAI_API_KEY="YOUR_API_KEY"
# ============================================================


def create_llm():

    task_started(
        "LLM Initialization"
    )

    start_time = time.time()

    try:

        task_info(
            "Initializing ChatOpenAI..."
        )

        llm = ChatOpenAI(

            model="gpt-4o-mini",

            temperature=0
        )

        elapsed_time = time.time() - start_time

        task_success(
            "LLM Initialization",
            f"LLM initialized in "
            f"{elapsed_time:.2f} seconds."
        )

        return llm

    except Exception as e:

        task_error(
            "LLM Initialization",
            e
        )

        raise


# ============================================================
# 11. CREATE RAG PROMPT
# ============================================================
# PURPOSE:
# Tell the LLM:
#
# - Use only retrieved PDF information.
# - Don't invent information.
# - Tell the user when information isn't available.
# ============================================================


def create_prompt():

    task_started(
        "RAG Prompt Creation"
    )

    try:

        prompt = ChatPromptTemplate.from_template(

            """
You are a helpful study assistant for the
AI-Empowered SAFe Product Owner/Product Manager course.

Answer the user's question using ONLY the context
provided below.

If the answer cannot be found in the context, say:

"I could not find this information in the uploaded PDF."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        task_success(
            "RAG Prompt Creation",
            "Prompt template created successfully."
        )

        return prompt

    except Exception as e:

        task_error(
            "RAG Prompt Creation",
            e
        )

        raise


# ============================================================
# 12. FORMAT RETRIEVED DOCUMENTS
# ============================================================
# PURPOSE:
# Convert retrieved LangChain Documents into text that can
# be passed to the LLM.
# ============================================================


def format_documents(documents):

    formatted_text = ""

    for document in documents:

        page_number = document.metadata.get(
            "page",
            "unknown"
        )

        if page_number != "unknown":

            page_number = page_number + 1

        formatted_text += (
            f"\n--- PDF Page {page_number} ---\n"
        )

        formatted_text += (
            document.page_content
        )

        formatted_text += "\n"

    return formatted_text


# ============================================================
# 13. ASK QUESTION
# ============================================================
# PURPOSE:
# Perform the complete RAG process.
#
# Question
#    ↓
# FAISS Similarity Search
#    ↓
# Relevant Documents
#    ↓
# Context
#    ↓
# Prompt
#    ↓
# LLM
#    ↓
# Answer
# ============================================================


def ask_question(
    vector_db,
    llm,
    prompt,
    question
):

    task_started(
        "Question Processing"
    )

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # STEP 1: SEARCH VECTOR DATABASE
        # ----------------------------------------------------

        task_info(
            "Searching FAISS for relevant information..."
        )

        retrieved_documents = (
            vector_db.similarity_search(
                question,
                k=4
            )
        )

        task_success(
            "FAISS Similarity Search",
            f"{len(retrieved_documents)} "
            f"relevant chunks retrieved."
        )

        # ----------------------------------------------------
        # STEP 2: CREATE CONTEXT
        # ----------------------------------------------------

        task_info(
            "Preparing context for the LLM..."
        )

        context = format_documents(
            retrieved_documents
        )

        task_success(
            "Context Preparation",
            "Retrieved document chunks "
            "converted into LLM context."
        )

        # ----------------------------------------------------
        # STEP 3: CREATE RAG CHAIN
        # ----------------------------------------------------

        task_info(
            "Creating LangChain RAG chain..."
        )

        chain = (

            prompt

            | llm

            | StrOutputParser()
        )

        task_success(
            "RAG Chain Creation",
            "Prompt → LLM → Output parser connected."
        )

        # ----------------------------------------------------
        # STEP 4: GENERATE ANSWER
        # ----------------------------------------------------

        task_info(
            "Sending context and question to LLM..."
        )

        answer = chain.invoke(

            {
                "context": context,

                "question": question
            }
        )

        elapsed_time = time.time() - start_time

        task_success(
            "Question Processing",
            f"Answer generated in "
            f"{elapsed_time:.2f} seconds."
        )

        return answer, retrieved_documents

    except Exception as e:

        task_error(
            "Question Processing",
            e
        )

        raise


# ============================================================
# 14. DISPLAY SOURCES
# ============================================================
# PURPOSE:
# Show the PDF pages from which information was retrieved.
#
# This helps you verify the RAG answer against the source PDF.
# ============================================================


def display_sources(
    documents
):

    task_started(
        "Source Identification"
    )

    try:

        pages = []

        for document in documents:

            page = document.metadata.get(
                "page",
                "unknown"
            )

            if page != "unknown":

                page_number = page + 1

            else:

                page_number = page

            if page_number not in pages:

                pages.append(
                    page_number
                )

        task_success(
            "Source Identification",
            f"Relevant PDF pages: {pages}"
        )

        return pages

    except Exception as e:

        task_error(
            "Source Identification",
            e
        )

        return []


# ============================================================
# 15. MAIN APPLICATION
# ============================================================
# PURPOSE:
# Start the complete RAG application.
# ============================================================


def main():

    print()
    print("#" * 70)
    print(
        " AI-EMPOWERED SAFE POPM - LOCAL RAG APPLICATION"
    )
    print("#" * 70)

    application_start = time.time()

    try:

        # ====================================================
        # TASK 1
        # PREPARE VECTOR DATABASE
        # ====================================================

        vector_db = get_vector_database()

        # ====================================================
        # TASK 2
        # INITIALIZE LLM
        # ====================================================

        llm = create_llm()

        # ====================================================
        # TASK 3
        # CREATE PROMPT
        # ====================================================

        prompt = create_prompt()

        # ====================================================
        # APPLICATION READY
        # ====================================================

        total_time = (
            time.time()
            - application_start
        )

        print()
        print("#" * 70)
        print(
            " SUCCESS: RAG APPLICATION IS READY"
        )
        print(
            f" Startup completed in {total_time:.2f} seconds."
        )
        print("#" * 70)

        print()
        print(
            "You can now ask questions about the PDF."
        )

        print(
            "Type 'exit' to stop the application."
        )

        print()
        print("-" * 70)

        # ====================================================
        # QUESTION / ANSWER LOOP
        # ====================================================

        while True:

            question = input(
                "\nYour Question: "
            ).strip()

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if question.lower() in [

                "exit",

                "quit",

                "q"

            ]:

                print()
                print(
                    "RAG application stopped successfully."
                )

                break

            # ------------------------------------------------
            # EMPTY QUESTION
            # ------------------------------------------------

            if not question:

                print(
                    "[INFO] Please enter a question."
                )

                continue

            # ------------------------------------------------
            # PROCESS QUESTION
            # ------------------------------------------------

            try:

                answer, sources = ask_question(

                    vector_db,

                    llm,

                    prompt,

                    question
                )

                # ------------------------------------------------
                # DISPLAY ANSWER
                # ------------------------------------------------

                print()
                print("#" * 70)
                print(
                    " ANSWER"
                )
                print("#" * 70)

                print(answer)

                # ------------------------------------------------
                # DISPLAY SOURCES
                # ------------------------------------------------

                display_sources(
                    sources
                )

                print()
                print(
                    "Question completed successfully."
                )

            except Exception as e:

                task_error(
                    "Question Answering",
                    e
                )

    except Exception as e:

        print()
        print("#" * 70)
        print(
            " APPLICATION FAILED"
        )
        print("#" * 70)

        print(
            f"Error: {e}"
        )


# ============================================================
# 16. RUN APPLICATION
# ============================================================
# PURPOSE:
# Start the application only when this Python file is executed.
# ============================================================


if __name__ == "__main__":

    main()