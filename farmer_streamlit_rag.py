# ============================================================
# FARMER SUPPORT RAG APPLICATION
# Vikaspedia + FAISS + Sentence Transformers + OpenAI + Streamlit
# ============================================================

import os
import re
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

import faiss
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# 2. APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "🌾 Farmer Support RAG Assistant"

SOURCE_URL = (
    "https://en.vikaspedia.in/viewcontent/"
    "schemesall/schemes-for-farmers"
)

SOURCE_DOMAINS = {
    "en.vikaspedia.in",
    "agriculture.vikaspedia.in",
    "schemes.vikaspedia.in",
}

DATA_FOLDER = Path("farmer_rag_data")

RAW_DOCUMENTS_FILE = DATA_FOLDER / "scraped_documents.json"
CHUNKS_FILE = DATA_FOLDER / "chunks.json"
FAISS_INDEX_FILE = DATA_FOLDER / "faiss_index.index"
METADATA_FILE = DATA_FOLDER / "metadata.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-5-mini"

TOP_K = 5
MAX_PAGES = 100
MAX_DEPTH = 2

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

REQUEST_TIMEOUT = 20
REQUEST_DELAY = 0.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    )
}


# ============================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Farmer Support RAG",
    page_icon="🌾",
    layout="wide",
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def show_success(message):
    st.success("✅ " + message)


def show_error(message):
    st.error("❌ " + message)


def show_info(message):
    st.info("ℹ️ " + message)


def normalize_url(url):
    parsed = urlparse(url)
    clean_url = parsed._replace(fragment="").geturl()
    return clean_url.rstrip("/")


def is_vikaspedia_url(url):
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ("http", "https")
            and parsed.netloc.lower() in SOURCE_DOMAINS
        )
    except Exception:
        return False


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 5. DOWNLOAD WEB PAGE
# ============================================================

def download_page(url):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

        print("Downloaded:", url)
        print("Final URL:", response.url)
        print("Status:", response.status_code)
        print("HTML length:", len(response.text))

        return response.text

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {url}")
        print(f"Error: {e}")

        raise Exception(
            f"Unable to download page: {url}\n{str(e)}"
        ) from e


# ============================================================
# 6. EXTRACT PAGE CONTENT
# ============================================================

def extract_page_content(url, html):
    soup = BeautifulSoup(html, "html.parser")

    for element in soup.find_all(
        ["script", "style", "noscript", "svg"]
    ):
        element.decompose()

    title = ""

    if soup.title:
        title = clean_text(
            soup.title.get_text(" ", strip=True)
        )

    headings = []

    for heading in soup.find_all(
        ["h1", "h2", "h3", "h4"]
    ):
        heading_text = clean_text(
            heading.get_text(" ", strip=True)
        )

        if heading_text:
            headings.append(heading_text)

    text = ""

    content_selectors = [
        "main",
        "article",
        "[role='main']",
        ".content",
        ".page-content",
        ".article-content",
        ".view-content",
    ]

    for selector in content_selectors:
        content = soup.select_one(selector)

        if content:
            extracted = clean_text(
                content.get_text(" ", strip=True)
            )

            if len(extracted) > len(text):
                text = extracted

    if len(text) < 100 and soup.body:
        text = clean_text(
            soup.body.get_text(" ", strip=True)
        )

    if len(text) < 100:
        text = clean_text(
            soup.get_text(" ", strip=True)
        )

    print(
        f"Extracted {len(text)} characters from {url}"
    )

    return {
        "url": url,
        "title": title,
        "headings": headings,
        "text": text,
    }


# ============================================================
# 7. FIND LINKS
# ============================================================

def extract_links(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")

        if not href:
            continue

        absolute_url = normalize_url(
            urljoin(base_url, href)
        )

        if is_vikaspedia_url(absolute_url):
            links.add(absolute_url)

    return links


# ============================================================
# 8. CHECK WHETHER PAGE IS RELEVANT
# ============================================================

def is_relevant_page(document):
    combined_text = (
        document.get("title", "")
        + " "
        + " ".join(document.get("headings", []))
        + " "
        + document.get("text", "")[:10000]
    ).lower()

    keywords = [
        "scheme",
        "farmer",
        "farmers",
        "agriculture",
        "agricultural",
        "horticulture",
        "farming",
        "crop",
        "livestock",
        "dairy",
        "poultry",
        "fisheries",
        "irrigation",
        "subsidy",
        "loan",
        "agri",
    ]

    matches = sum(
        1 for keyword in keywords
        if keyword in combined_text
    )

    return matches >= 2


# ============================================================
# 9. SCRAPE VIKASPEDIA
# ============================================================

def scrape_vikaspedia():
    documents = []
    visited = set()

    queue = [(SOURCE_URL, 0)]

    while queue and len(visited) < MAX_PAGES:
        current_url, depth = queue.pop(0)

        current_url = normalize_url(current_url)

        if current_url in visited:
            continue

        visited.add(current_url)

        try:
            html = download_page(current_url)

            document = extract_page_content(
                current_url,
                html,
            )

            print(f"URL: {current_url}")
            print(
                f"Title: {document.get('title', '')}"
            )
            print(
                "Text characters: "
                f"{len(document.get('text', ''))}"
            )

            if (
                current_url == normalize_url(SOURCE_URL)
                or is_relevant_page(document)
            ):
                documents.append(document)

            if depth >= MAX_DEPTH:
                continue

            links = extract_links(
                current_url,
                html,
            )

            for link in links:
                if link not in visited:
                    queue.append(
                        (link, depth + 1)
                    )

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(
                f"Skipping {current_url}: {e}"
            )

    return documents


# ============================================================
# 10. CREATE TEXT CHUNKS
# ============================================================

def create_chunks(documents):
    chunks = []
    chunk_id = 0

    for document in documents:
        text = document.get("text", "")

        if not text:
            continue

        title = document.get("title", "Unknown")
        url = document.get("url", "")

        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "title": title,
                        "url": url,
                        "text": chunk_text,
                    }
                )

                chunk_id += 1

            # IMPORTANT:
            # CHUNK_OVERLAP must be smaller than CHUNK_SIZE.
            start = end - CHUNK_OVERLAP

            if start <= 0:
                start = end

    return chunks


# ============================================================
# 11. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# 12. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks, model):
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    if not texts:
        raise ValueError(
            "No text is available for embedding."
        )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    return embeddings


# ============================================================
# 13. BUILD FAISS DATABASE
# ============================================================

def build_faiss_database(chunks, embeddings):
    if len(chunks) == 0:
        raise ValueError(
            "Cannot create FAISS database because "
            "there are no chunks."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    DATA_FOLDER.mkdir(
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE),
    )

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    metadata = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": dimension,
        "total_chunks": len(chunks),
        "source_url": SOURCE_URL,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return index


# ============================================================
# 14. BUILD COMPLETE KNOWLEDGE BASE
# ============================================================

def build_knowledge_base():
    """
    Build the complete local RAG knowledge base.

    IMPORTANT:
    The return statement is INSIDE this function.
    This fixes the original:
        SyntaxError: 'return' outside function
    """

    DATA_FOLDER.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    with st.status(
        "Stage 1: Preparing local RAG folder...",
        expanded=True,
    ) as status:

        try:
            DATA_FOLDER.mkdir(
                exist_ok=True
            )

            show_success(
                "Local RAG folder is ready."
            )

            status.update(
                label="Stage 1 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 1 failed",
                state="error",
            )
            raise e

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    with st.status(
        "Stage 2: Scraping Vikaspedia farmer schemes...",
        expanded=True,
    ) as status:

        try:
            st.write("Source:")
            st.code(SOURCE_URL)

            documents = scrape_vikaspedia()

            if not documents:
                raise Exception(
                    "No relevant Vikaspedia pages were found."
                )

            with open(
                RAW_DOCUMENTS_FILE,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    documents,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            show_success(
                "Scraping completed. "
                f"{len(documents)} relevant pages collected."
            )

            status.update(
                label="Stage 2 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 2 failed",
                state="error",
            )
            raise e

    # --------------------------------------------------------
    # STAGE 3
    # --------------------------------------------------------

    with st.status(
        "Stage 3: Creating text chunks...",
        expanded=True,
    ) as status:

        try:
            chunks = create_chunks(documents)

            if not chunks:
                total_documents = len(documents)

                documents_with_text = sum(
                    1
                    for doc in documents
                    if doc.get("text", "").strip()
                )

                raise Exception(
                    "No text chunks were created.\n\n"
                    f"Documents collected: {total_documents}\n"
                    f"Documents containing text: "
                    f"{documents_with_text}\n\n"
                    "The scraper downloaded pages but could not "
                    "extract usable text."
                )

            show_success(
                f"Created {len(chunks)} text chunks."
            )

            status.update(
                label="Stage 3 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 3 failed",
                state="error",
            )
            raise e

    # --------------------------------------------------------
    # STAGE 4
    # --------------------------------------------------------

    with st.status(
        "Stage 4: Loading embedding model...",
        expanded=True,
    ) as status:

        try:
            model = load_embedding_model()

            show_success(
                "Embedding model loaded: "
                f"{EMBEDDING_MODEL_NAME}"
            )

            status.update(
                label="Stage 4 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 4 failed",
                state="error",
            )
            raise e

    # --------------------------------------------------------
    # STAGE 5
    # --------------------------------------------------------

    with st.status(
        "Stage 5: Creating vector embeddings...",
        expanded=True,
    ) as status:

        try:
            embeddings = create_embeddings(
                chunks,
                model,
            )

            show_success(
                f"Created {len(embeddings)} embeddings."
            )

            status.update(
                label="Stage 5 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 5 failed",
                state="error",
            )
            raise e

    # --------------------------------------------------------
    # STAGE 6
    # --------------------------------------------------------

    with st.status(
        "Stage 6: Creating local FAISS database...",
        expanded=True,
    ) as status:

        try:
            index = build_faiss_database(
                chunks,
                embeddings,
            )

            show_success(
                "FAISS database created with "
                f"{index.ntotal} vectors."
            )

            status.update(
                label="Stage 6 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 6 failed",
                state="error",
            )
            raise e

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------
    # THIS MUST REMAIN INDENTED INSIDE build_knowledge_base().
    return documents, chunks, index


# ============================================================
# 15. LOAD EXISTING FAISS DATABASE
# ============================================================

def load_existing_database():
    if not (
        FAISS_INDEX_FILE.exists()
        and CHUNKS_FILE.exists()
    ):
        return None, None

    try:
        index = faiss.read_index(
            str(FAISS_INDEX_FILE)
        )

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            chunks = json.load(file)

        return index, chunks

    except Exception:
        return None, None


# ============================================================
# 16. SEARCH FAISS DATABASE
# ============================================================

def search_faiss(
    query,
    index,
    chunks,
    model,
    top_k=TOP_K,
):
    if index is None or index.ntotal == 0:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    faiss.normalize_L2(query_embedding)

    # Do not request more vectors than the index contains.
    actual_top_k = min(
        top_k,
        index.ntotal,
    )

    scores, indices = index.search(
        query_embedding,
        actual_top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):
        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        result = chunks[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# 17. GENERATE SEARCH QUERY
# ============================================================

def create_search_query(
    state,
    taluk,
    farming_type,
    requirement,
):
    query = f"""
Farmer is from State: {state}

Taluk: {taluk}

Farming type: {farming_type}

Farmer requirement:
{requirement}

Find relevant Indian agriculture,
farming, farmer welfare, subsidy, loan,
infrastructure, support and government
scheme information.
"""

    return query.strip()


# ============================================================
# 18. PREPARE CONTEXT FOR OPENAI
# ============================================================

def prepare_context(results):
    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):
        context_parts.append(
            f"""
SOURCE {i}

Title:
{result['title']}

URL:
{result['url']}

Similarity Score:
{result['score']:.4f}

Information:
{result['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# 19. GENERATE FINAL ANSWER USING OPENAI
# ============================================================

def generate_answer(
    farmer_name,
    state,
    taluk,
    farming_type,
    requirement,
    context,
):
    if not OPENAI_API_KEY:
        raise Exception(
            "OPENAI_API_KEY was not found.\n\n"
            "Please check your .env file."
        )

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    prompt = f"""
You are an AI Farmer Support Assistant.

Your job is to help farmers understand relevant
government agriculture schemes and support information.

IMPORTANT RULES:

1. Use ONLY the Vikaspedia information supplied
   in the context below.

2. Do NOT invent any scheme.

3. Do NOT invent subsidy amounts.

4. Do NOT invent eligibility criteria.

5. Do NOT invent application procedures.

6. Do NOT assume that the farmer is eligible.

7. If the retrieved information does not answer
   the farmer's question, clearly say that the
   available Vikaspedia information is insufficient.

8. Give the answer in simple language.

9. Clearly identify the relevant scheme or support.

10. Mention important conditions if they are present
    in the retrieved information.

11. If state-specific information is not available,
    clearly tell the farmer to verify with the
    appropriate government department.

FARMER DETAILS

Name:
{farmer_name}

State:
{state}

Taluk:
{taluk}

Farming Type:
{farming_type}

Requirement:
{requirement}

RETRIEVED VIKASPEDIA INFORMATION

{context}

ANSWER FORMAT

Please provide:

1. Suitable Scheme / Support
2. Why it may be relevant
3. Main benefits
4. Eligibility / conditions
5. How to apply, if available
6. Important documents, if available
7. Important notes
8. Source information

Do not provide information that is not supported
by the retrieved Vikaspedia content.
"""

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
    )

    return response.output_text


# ============================================================
# 20. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ RAG Controls")

st.sidebar.write(
    "Use the button below to scrape Vikaspedia "
    "and rebuild the local FAISS database."
)

rebuild_database = st.sidebar.button(
    "🔄 Scrape & Rebuild FAISS",
    use_container_width=True,
)

st.sidebar.markdown("---")

st.sidebar.write("**Source:** Vikaspedia")
st.sidebar.write(
    f"**Embedding:** {EMBEDDING_MODEL_NAME}"
)
st.sidebar.write(f"**LLM:** {LLM_MODEL}")
st.sidebar.write(f"**Top K:** {TOP_K}")


# ============================================================
# 21. APPLICATION TITLE
# ============================================================

st.title(APP_TITLE)

st.subheader(
    "AI-powered agriculture scheme and support assistant"
)

st.write(
    "This application retrieves relevant information "
    "from Vikaspedia using a local FAISS vector database "
    "and generates a farmer-friendly answer using OpenAI."
)

st.markdown("---")


# ============================================================
# 22. DISPLAY DATABASE STATUS
# ============================================================

index, chunks = load_existing_database()

if index is not None:
    st.success(
        "🟢 Local FAISS database available — "
        f"{index.ntotal} vectors."
    )
else:
    st.warning(
        "🟡 FAISS database is not available. "
        "Use 'Scrape & Rebuild FAISS' from the sidebar."
    )


# ============================================================
# 23. REBUILD DATABASE
# ============================================================

if rebuild_database:
    st.markdown("## 🔄 Knowledge Base Processing")

    try:
        documents, chunks, index = build_knowledge_base()

        st.success(
            "🎉 Knowledge base successfully created!"
        )

        st.write(
            f"Pages collected: {len(documents)}"
        )
        st.write(
            f"Text chunks created: {len(chunks)}"
        )
        st.write(
            f"FAISS vectors: {index.ntotal}"
        )

        st.session_state["index"] = index
        st.session_state["chunks"] = chunks

    except Exception as e:
        st.error(
            "Knowledge base creation failed:\n\n"
            f"{str(e)}"
        )


# ============================================================
# 24. FARMER INFORMATION FORM
# ============================================================

st.markdown("## 👨‍🌾 Farmer Information")

with st.form("farmer_form"):
    farmer_name = st.text_input(
        "Farmer Name",
        placeholder="Enter farmer name",
    )

    col1, col2 = st.columns(2)

    with col1:
        state = st.text_input(
            "State",
            placeholder="Example: Tamil Nadu",
        )

    with col2:
        taluk = st.text_input(
            "Taluk",
            placeholder="Example: Madurai North",
        )

    farming_type = st.selectbox(
        "Type of Farming",
        [
            "Crop Farming",
            "Horticulture",
            "Organic Farming",
            "Dairy Farming",
            "Poultry Farming",
            "Fisheries",
            "Animal Husbandry",
            "Sericulture",
            "Mixed Farming",
            "Other",
        ],
    )

    requirement = st.text_area(
        "What information or support do you need?",
        placeholder=(
            "Example:\n"
            "I need information about government "
            "schemes for irrigation support."
        ),
        height=150,
    )

    submit = st.form_submit_button(
        "🔍 Find Relevant Schemes",
        use_container_width=True,
    )


# ============================================================
# 25. PROCESS FARMER REQUEST
# ============================================================

if submit:

    # --------------------------------------------------------
    # STAGE 1: VALIDATION
    # --------------------------------------------------------

    with st.status(
        "Stage 1: Validating farmer information...",
        expanded=True,
    ) as status:

        try:
            if not farmer_name.strip():
                raise Exception(
                    "Please enter farmer name."
                )

            if not state.strip():
                raise Exception(
                    "Please enter state."
                )

            if not taluk.strip():
                raise Exception(
                    "Please enter taluk."
                )

            if not requirement.strip():
                raise Exception(
                    "Please enter your requirement."
                )

            show_success(
                "Farmer information is valid."
            )

            status.update(
                label="Stage 1 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 1 failed",
                state="error",
            )
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # STAGE 2: CHECK DATABASE
    # --------------------------------------------------------

    with st.status(
        "Stage 2: Loading local FAISS database...",
        expanded=True,
    ) as status:

        try:
            index, chunks = load_existing_database()

            if index is None:
                raise Exception(
                    "FAISS database is not available. "
                    "Please click 'Scrape & Rebuild FAISS' "
                    "from the sidebar first."
                )

            show_success(
                "FAISS database loaded. "
                f"{index.ntotal} vectors available."
            )

            status.update(
                label="Stage 2 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 2 failed",
                state="error",
            )
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # STAGE 3: LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    with st.status(
        "Stage 3: Loading embedding model...",
        expanded=True,
    ) as status:

        try:
            embedding_model = load_embedding_model()

            show_success(
                "Embedding model loaded."
            )

            status.update(
                label="Stage 3 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 3 failed",
                state="error",
            )
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # STAGE 4: CREATE SEMANTIC QUERY
    # --------------------------------------------------------

    with st.status(
        "Stage 4: Creating semantic search query...",
        expanded=True,
    ) as status:

        try:
            search_query = create_search_query(
                state,
                taluk,
                farming_type,
                requirement,
            )

            st.write("Search query:")
            st.code(search_query)

            show_success(
                "Semantic search query created."
            )

            status.update(
                label="Stage 4 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 4 failed",
                state="error",
            )
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # STAGE 5: RETRIEVE INFORMATION
    # --------------------------------------------------------

    with st.status(
        "Stage 5: Searching FAISS for relevant information...",
        expanded=True,
    ) as status:

        try:
            results = search_faiss(
                search_query,
                index,
                chunks,
                embedding_model,
                TOP_K,
            )

            if not results:
                raise Exception(
                    "No relevant information found."
                )

            show_success(
                f"Retrieved {len(results)} relevant "
                "knowledge chunks."
            )

            status.update(
                label="Stage 5 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 5 failed",
                state="error",
            )
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # STAGE 6: SHOW RETRIEVED RAG INFORMATION
    # --------------------------------------------------------

    with st.status(
        "Stage 6: Preparing retrieved RAG information...",
        expanded=False,
    ) as status:

        try:
            st.markdown(
                "### 📚 Retrieved RAG Information"
            )

            for i, result in enumerate(
                results,
                start=1,
            ):
                with st.expander(
                    f"Source {i}: {result['title']}"
                ):
                    st.write(
                        "**Similarity Score:** "
                        f"{result['score']:.4f}"
                    )
                    st.write(
                        f"**URL:** {result['url']}"
                    )
                    st.write(result["text"])

            show_success(
                "Retrieved information prepared."
            )

            status.update(
                label="Stage 6 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 6 failed",
                state="error",
            )
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # STAGE 7: CREATE CONTEXT
    # --------------------------------------------------------

    with st.status(
        "Stage 7: Creating context for the LLM...",
        expanded=True,
    ) as status:

        try:
            context = prepare_context(results)

            show_success(
                "Retrieved context prepared for OpenAI."
            )

            status.update(
                label="Stage 7 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 7 failed",
                state="error",
            )
            st.error(str(e))
            st.stop()

    # --------------------------------------------------------
    # STAGE 8: OPENAI ANSWER GENERATION
    # --------------------------------------------------------

    with st.status(
        "Stage 8: Generating farmer-friendly answer...",
        expanded=True,
    ) as status:

        try:
            answer = generate_answer(
                farmer_name=farmer_name,
                state=state,
                taluk=taluk,
                farming_type=farming_type,
                requirement=requirement,
                context=context,
            )

            show_success(
                "OpenAI generated the final answer."
            )

            status.update(
                label="Stage 8 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 8 failed",
                state="error",
            )
            st.error(
                "OpenAI answer generation failed:\n\n"
                f"{str(e)}"
            )
            st.stop()

    # --------------------------------------------------------
    # STAGE 9: DISPLAY FINAL ANSWER
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "## 🌾 Farmer Support Recommendation"
    )

    st.write(
        f"**Farmer:** {farmer_name}"
    )

    st.write(
        f"**State:** {state}"
    )

    st.write(
        f"**Taluk:** {taluk}"
    )

    st.write(
        f"**Farming Type:** {farming_type}"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # FINAL AI ANSWER - ALWAYS VISIBLE
    # --------------------------------------------------------

    st.markdown(
        "## 🤖 AI Answer"
    )

    # Display the OpenAI generated answer directly
    # without placing it inside st.status()
    st.markdown(answer)

    st.success(
        "✅ Stage 9 completed — Final answer displayed successfully."
    )

    # --------------------------------------------------------
    # STAGE 10: DISPLAY SOURCES
    # --------------------------------------------------------

    with st.status(
        "Stage 10: Preparing source references...",
        expanded=False,
    ) as status:

        try:
            st.markdown(
                "### 🔗 Vikaspedia Sources"
            )

            unique_urls = []

            for result in results:
                url = result["url"]

                if url not in unique_urls:
                    unique_urls.append(url)

            for url in unique_urls:
                st.markdown(
                    f"- [{url}]({url})"
                )

            show_success(
                f"{len(unique_urls)} source pages identified."
            )

            status.update(
                label="Stage 10 completed",
                state="complete",
            )

        except Exception as e:
            status.update(
                label="Stage 10 failed",
                state="error",
            )
            st.error(str(e))


# ============================================================
# 26. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🌾 Farmer Support RAG | "
    "Knowledge source: Vikaspedia | "
    "Vector database: FAISS | "
    "Embeddings: Sentence Transformers | "
    "LLM: OpenAI"
)
