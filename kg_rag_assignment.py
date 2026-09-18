# ============================================================
# KG-RAG ASSIGNMENT
# LangChain + OpenAI + Neo4j AuraDB
#
# INPUT:
#   Put one or more website URLs / document paths in
#   input_sources.txt
#
# RUN:
#   python kg_rag_assignment.py
#
# The program will:
#   1. Read URLs / document paths from input_sources.txt
#   2. Load the source content
#   3. Extract entities with OpenAI + LangChain
#   4. Extract relationships with OpenAI + LangChain
#   5. Store the Knowledge Graph in Neo4j AuraDB
#   6. Retrieve relevant graph information
#   7. Generate an answer with OpenAI
#   8. Continue asking questions until q is entered
# ============================================================

import os
import re
import ssl
from pathlib import Path
from typing import List

import certifi
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from neo4j import GraphDatabase

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
INPUT_FILE = BASE_DIR / "input_sources.txt"

load_dotenv(ENV_PATH)

MODEL_NAME = "gpt-5-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


# ============================================================
# HELPER: STATUS
# ============================================================

def print_stage(number, title):
    print("\n" + "=" * 70)
    print(f"[{number}] {title}")
    print("=" * 70)


# ============================================================
# STEP 1: ENVIRONMENT VALIDATION
# ============================================================

def validate_environment():
    print_stage(1, "ENVIRONMENT CHECK")

    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not NEO4J_URI:
        missing.append("NEO4J_URI")

    if not NEO4J_USERNAME:
        missing.append("NEO4J_USERNAME")

    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")

    if not NEO4J_DATABASE:
        missing.append("NEO4J_DATABASE")

    if missing:
        print("✗ Missing environment variables:")
        for item in missing:
            print(f"  - {item}")
        return False

    print("✓ .env file loaded")
    print("✓ OpenAI API key found")
    print("✓ Neo4j URI found")
    print("✓ Neo4j username found")
    print("✓ Neo4j password found")
    print(f"✓ Neo4j database: {NEO4J_DATABASE}")

    return True


# ============================================================
# STEP 2: READ INPUT FILE
# ============================================================

def read_input_sources():
    print_stage(2, "READING INPUT SOURCES")

    if not INPUT_FILE.exists():
        print(f"✗ Input file not found:")
        print(f"  {INPUT_FILE}")
        print("\nCreate input_sources.txt in the same folder as this program.")
        return []

    lines = INPUT_FILE.read_text(encoding="utf-8").splitlines()

    sources = []

    for line in lines:
        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        # Ignore comments beginning with #
        if line.startswith("#"):
            continue

        sources.append(line)

    if not sources:
        print("✗ No URLs or document paths found in input_sources.txt")
        return []

    print(f"✓ Found {len(sources)} source(s)")

    for index, source in enumerate(sources, start=1):
        print(f"  {index}. {source}")

    return sources


# ============================================================
# STEP 3: DETECT SOURCE TYPE
# ============================================================

def detect_input_type(source):
    source_lower = source.lower().strip()

    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return "url"

    path = Path(source)

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        return "unknown"

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return "txt"

    if suffix == ".pdf":
        return "pdf"

    if suffix == ".docx":
        return "docx"

    return "unknown"


# ============================================================
# STEP 4: CLEAN WEBPAGE TEXT
# ============================================================

def clean_web_text(text):
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove obvious repeated separators
        if re.fullmatch(r"[-_=]{3,}", line):
            continue

        lines.append(line)

    text = "\n".join(lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# STEP 5: WEBSITE LOADER
#
# This loader is deliberately more careful than a simple
# soup.get_text() call.
#
# It:
#   - requests the page
#   - removes navigation/script/style elements
#   - looks for main/article/table content
#   - preserves headings and table information
#   - checks whether useful content was actually obtained
# ============================================================

def load_website(url):
    print(f"\nLoading website:")
    print(f"  {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
        allow_redirects=True
    )

    response.raise_for_status()

    print(f"✓ HTTP status: {response.status_code}")
    print(f"✓ Final URL: {response.url}")
    print(f"✓ Downloaded: {len(response.text):,} HTML characters")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove elements that normally contain navigation,
    # scripts, styling, or unrelated page controls.
    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
    ]):
        tag.decompose()

    content_parts = []

    # --------------------------------------------------------
    # Prefer main/article content when available.
    # --------------------------------------------------------

    candidates = []

    for selector in [
        "main",
        "article",
        '[role="main"]',
        ".main-content",
        ".content",
        ".container",
        "#content",
        "#main",
    ]:
        candidates.extend(soup.select(selector))

    # Remove duplicate candidate objects while preserving order.
    unique_candidates = []
    seen_ids = set()

    for candidate in candidates:
        marker = id(candidate)
        if marker not in seen_ids:
            unique_candidates.append(candidate)
            seen_ids.add(marker)

    # --------------------------------------------------------
    # Extract headings.
    # --------------------------------------------------------

    if unique_candidates:
        roots = unique_candidates
    else:
        roots = [soup.body if soup.body else soup]

    for root in roots:
        for element in root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6"]
        ):
            value = element.get_text(" ", strip=True)
            if value:
                content_parts.append(value)

    # --------------------------------------------------------
    # Extract tables.
    #
    # Tables are important for government scheme pages because
    # scheme names/details may be represented in table cells.
    # --------------------------------------------------------

    for table in soup.find_all("table"):
        rows = []

        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])

            values = [
                cell.get_text(" ", strip=True)
                for cell in cells
            ]

            values = [value for value in values if value]

            if values:
                rows.append(" | ".join(values))

        if rows:
            content_parts.append("\n".join(rows))

    # --------------------------------------------------------
    # Extract paragraphs and list items.
    # --------------------------------------------------------

    for root in roots:
        for element in root.find_all(["p", "li", "dt", "dd"]):
            value = element.get_text(" ", strip=True)

            if value and len(value) > 2:
                content_parts.append(value)

    # --------------------------------------------------------
    # Fallback: if targeted extraction was too small, use the
    # complete visible body text after unwanted elements were
    # removed.
    # --------------------------------------------------------

    targeted_text = clean_web_text("\n".join(content_parts))

    body_text = ""

    if soup.body:
        body_text = clean_web_text(
            soup.body.get_text("\n", strip=True)
        )

    if len(targeted_text) >= 300:
        final_text = targeted_text
    else:
        final_text = body_text

    print(f"✓ Extracted useful text: {len(final_text):,} characters")

    # --------------------------------------------------------
    # Diagnostic preview.
    # --------------------------------------------------------

    preview = final_text[:1000].replace("\n", " ")

    if preview:
        print("\nWebpage text preview:")
        print("-" * 70)
        print(preview)
        print("-" * 70)

    if len(final_text) < 100:
        print("\n⚠ Very little text was extracted from this webpage.")
        print("  The site may require JavaScript or another extraction method.")

    return final_text


# ============================================================
# STEP 6: DOCUMENT LOADER
# ============================================================

def load_document(source):
    path = Path(source)

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))

    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))

    else:
        raise ValueError(
            "Supported document types are .txt, .pdf and .docx"
        )

    documents = loader.load()

    text = "\n\n".join(
        document.page_content
        for document in documents
    )

    text = clean_web_text(text)

    print(f"✓ Document loaded: {path}")
    print(f"✓ Extracted text: {len(text):,} characters")

    return text


# ============================================================
# STEP 7: LOAD ONE SOURCE
# ============================================================

def load_source(source):
    source_type = detect_input_type(source)

    print(f"\nSource type: {source_type}")

    if source_type == "url":
        return load_website(source)

    if source_type in ["txt", "pdf", "docx"]:
        return load_document(source)

    raise ValueError(
        f"Unsupported or missing source: {source}"
    )


# ============================================================
# STEP 8: CREATE OPENAI MODEL
# ============================================================

def create_llm():
    print("\nCreating OpenAI model...")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    print(f"✓ ChatOpenAI created: {MODEL_NAME}")

    return llm


# ============================================================
# STEP 9: PYDANTIC SCHEMAS
# ============================================================

class Entity(BaseModel):
    name: str = Field(description="Name of the entity")
    entity_type: str = Field(
        description="Type/category of the entity"
    )
    description: str = Field(
        description="Short description of the entity"
    )


class EntityExtraction(BaseModel):
    entities: List[Entity]


class Relationship(BaseModel):
    source: str = Field(
        description="Source entity name"
    )
    relationship: str = Field(
        description="Relationship between source and target"
    )
    target: str = Field(
        description="Target entity name"
    )
    description: str = Field(
        description="Short explanation of the relationship"
    )


class RelationshipExtraction(BaseModel):
    relationships: List[Relationship]


# ============================================================
# STEP 10: ENTITY EXTRACTION
# ============================================================

def extract_entities(llm, text):
    print_stage(3, "ENTITY EXTRACTION")

    if not text.strip():
        print("✗ No source text available")
        return []

    # Limit the size of one extraction request.
    # This keeps the beginner project simple and avoids
    # unnecessarily large prompts.
    max_chars = 50000
    working_text = text[:max_chars]

    structured_llm = llm.with_structured_output(
        EntityExtraction,
        method="json_schema"
    )

    prompt = f"""
You are extracting entities for a Knowledge Graph.

Extract important factual entities from the source text.

Focus especially on:
- Government departments
- Government schemes
- Technologies
- Agricultural technologies
- Crops
- Organizations
- Locations
- People
- Products
- Funds
- Services
- Beneficiary groups
- Important concepts

Do NOT invent entities.

Use the wording from the source whenever possible.

SOURCE TEXT:
{working_text}
"""

    result = structured_llm.invoke(prompt)

    entities = result.entities

    print(f"✓ Entities extracted: {len(entities)}")

    for entity in entities[:20]:
        print(
            f"  - {entity.name} "
            f"[{entity.entity_type}]"
        )

    if len(entities) > 20:
        print(f"  ... and {len(entities) - 20} more")

    return entities


# ============================================================
# STEP 11: RELATIONSHIP EXTRACTION
# ============================================================

def extract_relationships(llm, text, entities):
    print_stage(4, "RELATIONSHIP EXTRACTION")

    if not entities:
        print("⚠ No entities available")
        return []

    entity_text = "\n".join(
        f"- {entity.name} [{entity.entity_type}]"
        for entity in entities
    )

    max_chars = 50000
    working_text = text[:max_chars]

    structured_llm = llm.with_structured_output(
        RelationshipExtraction,
        method="json_schema"
    )

    prompt = f"""
You are extracting relationships for a Knowledge Graph.

Use ONLY relationships supported by the source text.

Do NOT invent relationships.

Known entities:
{entity_text}

SOURCE TEXT:
{working_text}

For each relationship, provide:
- source
- relationship
- target
- description

Use concise relationship names such as:
OFFERS
IMPLEMENTS
SUPPORTS
BENEFITS
LOCATED_IN
RELATED_TO
PROVIDES
TARGETS
USES
APPLIES_TO
"""

    result = structured_llm.invoke(prompt)

    relationships = result.relationships

    print(f"✓ Relationships extracted: {len(relationships)}")

    for relationship in relationships[:20]:
        print(
            f"  - {relationship.source} "
            f"--[{relationship.relationship}]--> "
            f"{relationship.target}"
        )

    if len(relationships) > 20:
        print(
            f"  ... and {len(relationships) - 20} more"
        )

    return relationships


# ============================================================
# STEP 12: NEO4J CONNECTION
#
# IMPORTANT:
# We use neo4j:// routing rather than direct bolt://.
# This is the method successfully tested with the user's
# AuraDB instance.
# ============================================================

def connect_to_neo4j():
    print_stage(5, "CONNECTING TO NEO4J AURADB")

    try:
        routing_uri = NEO4J_URI.replace(
            "neo4j+s://",
            "neo4j://"
        )

        print(f"Neo4j URI      : {routing_uri}")
        print(f"Neo4j username : {NEO4J_USERNAME}")
        print(f"Neo4j database : {NEO4J_DATABASE}")

        ssl_context = ssl.create_default_context(
            cafile=certifi.where()
        )

        driver = GraphDatabase.driver(
            routing_uri,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            encrypted=True,
            ssl_context=ssl_context
        )

        print("\n✓ Neo4j driver created")

        driver.verify_connectivity()

        print("✓ Neo4j connectivity successful")

        result = driver.execute_query(
            "RETURN 1 AS test",
            database_=NEO4J_DATABASE
        )

        for record in result.records:
            print(
                f"✓ Database test successful: "
                f"{record['test']}"
            )

        print("✓ Connected successfully to Neo4j AuraDB")

        return driver

    except Exception as error:
        print("\n✗ Neo4j connection failed")
        print(error)
        return None


# ============================================================
# STEP 13: CLEAN RELATIONSHIP TYPE
# ============================================================

def clean_relationship_type(value):
    value = value.upper().strip()

    value = re.sub(
        r"[^A-Z0-9_]+",
        "_",
        value
    )

    value = re.sub(
        r"_+",
        "_",
        value
    )

    value = value.strip("_")

    if not value:
        value = "RELATED_TO"

    return value[:80]


# ============================================================
# STEP 14: STORE KNOWLEDGE GRAPH
# ============================================================

def store_knowledge_graph(
    driver,
    source,
    entities,
    relationships
):
    print_stage(6, "GRAPH STORAGE")

    source_name = source[:500]

    # --------------------------------------------------------
    # Create Source node
    # --------------------------------------------------------

    driver.execute_query(
        """
        MERGE (s:Source {name: $source})
        SET s.source_type = $source_type
        """,
        source=source_name,
        source_type=detect_input_type(source),
        database_=NEO4J_DATABASE
    )

    # --------------------------------------------------------
    # Create Entity nodes
    # --------------------------------------------------------

    for entity in entities:

        driver.execute_query(
            """
            MERGE (e:Entity {name: $name})
            SET e.entity_type = $entity_type,
                e.description = $description
            """,
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            database_=NEO4J_DATABASE
        )

        driver.execute_query(
            """
            MATCH (s:Source {name: $source})
            MATCH (e:Entity {name: $name})
            MERGE (s)-[:CONTAINS]->(e)
            """,
            source=source_name,
            name=entity.name,
            database_=NEO4J_DATABASE
        )

    # --------------------------------------------------------
    # Create relationships
    # --------------------------------------------------------

    for relationship in relationships:

        rel_type = clean_relationship_type(
            relationship.relationship
        )

        query = f"""
        MATCH (a:Entity {{name: $source}})
        MATCH (b:Entity {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.description = $description
        """

        driver.execute_query(
            query,
            source=relationship.source,
            target=relationship.target,
            description=relationship.description,
            database_=NEO4J_DATABASE
        )

    print(
        f"✓ Stored {len(entities)} entities "
        f"and {len(relationships)} relationships"
    )


# ============================================================
# STEP 15: GRAPH SUMMARY
# ============================================================

def get_graph_summary(driver):
    print_stage(7, "GRAPH SUMMARY")

    entity_result = driver.execute_query(
        "MATCH (e:Entity) RETURN count(e) AS count",
        database_=NEO4J_DATABASE
    )

    relationship_result = driver.execute_query(
        """
        MATCH ()-[r]->()
        RETURN count(r) AS count
        """,
        database_=NEO4J_DATABASE
    )

    entity_count = entity_result.records[0]["count"]
    relationship_count = relationship_result.records[0]["count"]

    print(f"✓ Total Entity nodes       : {entity_count}")
    print(f"✓ Total relationships      : {relationship_count}")


# ============================================================
# STEP 16: QUESTION KEYWORDS
# ============================================================

def get_question_keywords(question):
    stop_words = {
        "what",
        "what's",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "and",
        "or",
        "does",
        "do",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "can",
        "about",
        "tell",
        "me",
        "please",
        "explain",
        "give",
        "information",
        "informationabout",
    }

    words = re.findall(
        r"[A-Za-z0-9]+",
        question.lower()
    )

    keywords = [
        word
        for word in words
        if word not in stop_words and len(word) >= 3
    ]

    return keywords


# ============================================================
# STEP 17: IMPROVED GRAPH RETRIEVAL
#
# The previous version could match the word "technology" and
# return an unrelated entity such as "Technology Development
# Fund".
#
# This version gives more importance to:
#   - exact phrase matches
#   - exact entity name matches
#   - multiple keyword matches
#   - entity names containing multiple question terms
#
# It also avoids returning a single generic "technology"
# match when a more specific match exists.
# ============================================================

def retrieve_graph_context(driver, question):
    print_stage(8, "KG-RAG RETRIEVAL")

    print("\nSearching Knowledge Graph...")

    keywords = get_question_keywords(question)

    if not keywords:
        keywords = [question.lower().strip()]

    print(f"Question keywords: {keywords}")

    question_text = question.lower().strip()

    # --------------------------------------------------------
    # Search entity names and descriptions.
    # --------------------------------------------------------

    entity_query = """
    MATCH (e:Entity)

    WITH e,
         toLower(e.name) AS entity_name,
         toLower(coalesce(e.description, "")) AS entity_description

    WITH e,
         entity_name,
         entity_description,
         reduce(
             score = 0,
             keyword IN $keywords |
             score +
             CASE
                 WHEN entity_name = keyword
                 THEN 12

                 WHEN entity_name CONTAINS keyword
                 THEN 4

                 WHEN entity_description CONTAINS keyword
                 THEN 1

                 ELSE 0
             END
         ) AS keyword_score

    RETURN
        e.name AS name,
        e.entity_type AS entity_type,
        e.description AS description,
        keyword_score

    ORDER BY keyword_score DESC, name
    LIMIT 30
    """

    result = driver.execute_query(
        entity_query,
        keywords=keywords,
        database_=NEO4J_DATABASE
    )

    candidates = [
        record
        for record in result.records
        if record["keyword_score"] > 0
    ]

    # --------------------------------------------------------
    # Additional Python-side scoring.
    #
    # This gives a large bonus when multiple words from the
    # question occur in the SAME entity name.
    # --------------------------------------------------------

    scored = []

    for record in candidates:

        name_lower = record["name"].lower()

        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword in name_lower
        ]

        score = record["keyword_score"]

        # Multiple keywords in the same entity are stronger
        # evidence than a single generic keyword.
        score += len(matched_keywords) * 3

        # Exact phrase match is especially strong.
        phrase = question_text

        if phrase in name_lower:
            score += 30

        # Check a compact phrase without common question words.
        compact_phrase = " ".join(keywords)

        if compact_phrase in name_lower:
            score += 20

        scored.append(
            (
                score,
                record,
                matched_keywords
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            len(item[2])
        ),
        reverse=True
    )

    top_entities = scored[:10]

    print(
        f"✓ Relevant entities found: "
        f"{len(top_entities)}"
    )

    if top_entities:
        print("\nMatched entities:")

        for score, record, matched_keywords in top_entities:
            print(
                f"  - {record['name']} "
                f"[{record['entity_type']}] "
                f"(score: {score}, "
                f"matched: {matched_keywords})"
            )

    if not top_entities:
        print("⚠ No matching entities found.")
        return ""

    entity_names = [
        record["name"]
        for score, record, matched_keywords
        in top_entities
    ]

    # --------------------------------------------------------
    # Retrieve relationships connected to the best entities.
    # --------------------------------------------------------

    relationship_query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE a.name IN $entity_names
       OR b.name IN $entity_names

    RETURN
        a.name AS source,
        type(r) AS relationship,
        b.name AS target,
        coalesce(r.description, "") AS description

    LIMIT 50
    """

    relationship_result = driver.execute_query(
        relationship_query,
        entity_names=entity_names,
        database_=NEO4J_DATABASE
    )

    relationships = relationship_result.records

    print(
        f"✓ Related relationships found: "
        f"{len(relationships)}"
    )

    # --------------------------------------------------------
    # Build focused context.
    # --------------------------------------------------------

    context_parts = []

    context_parts.append("RELEVANT ENTITIES:")

    for score, record, matched_keywords in top_entities:
        context_parts.append(
            f"- {record['name']} "
            f"[{record['entity_type']}]: "
            f"{record['description']}"
        )

    context_parts.append("")
    context_parts.append("RELEVANT RELATIONSHIPS:")

    for record in relationships:
        context_parts.append(
            f"- {record['source']} "
            f"--[{record['relationship']}]--> "
            f"{record['target']}: "
            f"{record['description']}"
        )

    context = "\n".join(context_parts)

    print("\n✓ Focused Knowledge Graph context created.")
    print("✓ Relevant graph information found.")

    return context


# ============================================================
# STEP 18: GENERATE FINAL ANSWER
# ============================================================

def generate_answer(llm, question, graph_context):
    print_stage(9, "OPENAI ANSWER GENERATION")

    print("\nGenerating answer with OpenAI...")

    if not graph_context:
        graph_context = (
            "No relevant information was found "
            "in the Knowledge Graph."
        )

    prompt = f"""
You are answering a question using a Knowledge Graph.

IMPORTANT RULES:

1. Use the Knowledge Graph context as the primary source.
2. Do not invent facts.
3. If the KG does not contain enough information, clearly say so.
4. Do not claim that an unrelated entity answers the question.
5. Distinguish between information present in the KG and information
   that is not present.
6. Give a clear beginner-friendly answer.

USER QUESTION:
{question}

KNOWLEDGE GRAPH CONTEXT:
{graph_context}
"""

    response = llm.invoke(prompt)

    answer = response.content

    print("\n" + "=" * 70)
    print("FINAL KG-RAG ANSWER")
    print("=" * 70)

    print(answer)

    print("=" * 70)

    return answer


# ============================================================
# STEP 19: INTERACTIVE QUESTION LOOP
# ============================================================

def interactive_question_loop(driver, llm):
    print("\n" + "=" * 70)
    print("KG-RAG READY")
    print("=" * 70)

    print("The Knowledge Graph is ready.")
    print("\nAsk questions about the loaded information.")
    print("Type 'q' to safely quit.\n")

    while True:

        try:
            question = input("\nAsk your question: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting KG-RAG safely...")
            break

        if question.lower() == "q":
            print("\nExiting KG-RAG safely...")
            break

        if not question:
            print("Please enter a question.")
            continue

        graph_context = retrieve_graph_context(
            driver,
            question
        )

        generate_answer(
            llm,
            question,
            graph_context
        )


# ============================================================
# STEP 20: MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("KG-RAG ASSIGNMENT")
    print("=" * 70)

    print("LangChain + OpenAI + Neo4j AuraDB")

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    if not validate_environment():
        return

    # --------------------------------------------------------
    # Read URLs / files
    # --------------------------------------------------------

    sources = read_input_sources()

    if not sources:
        return

    # --------------------------------------------------------
    # Create OpenAI model
    # --------------------------------------------------------

    llm = create_llm()

    # --------------------------------------------------------
    # Connect to Neo4j
    # --------------------------------------------------------

    driver = connect_to_neo4j()

    if driver is None:
        print("\n✗ Cannot continue without Neo4j.")
        return

    try:

        # ----------------------------------------------------
        # Process every source from input_sources.txt
        # ----------------------------------------------------

        for source_number, source in enumerate(
            sources,
            start=1
        ):

            print_stage(
                f"3.{source_number}",
                f"PROCESSING SOURCE {source_number}"
            )

            print(f"\nSource:")
            print(source)

            try:

                # --------------------------------------------
                # Load source
                # --------------------------------------------

                text = load_source(source)

                if not text.strip():
                    print(
                        "✗ No usable text extracted. "
                        "Skipping this source."
                    )
                    continue

                # --------------------------------------------
                # Entity extraction
                # --------------------------------------------

                entities = extract_entities(
                    llm,
                    text
                )

                # --------------------------------------------
                # Relationship extraction
                # --------------------------------------------

                relationships = extract_relationships(
                    llm,
                    text,
                    entities
                )

                # --------------------------------------------
                # Graph storage
                # --------------------------------------------

                store_knowledge_graph(
                    driver,
                    source,
                    entities,
                    relationships
                )

                print(
                    f"\n✓ Source {source_number} "
                    f"processed successfully."
                )

            except Exception as error:

                print(
                    f"\n✗ Error processing source "
                    f"{source_number}:"
                )
                print(error)

        # ----------------------------------------------------
        # Graph summary
        # ----------------------------------------------------

        get_graph_summary(driver)

        # ----------------------------------------------------
        # Interactive KG-RAG
        # ----------------------------------------------------

        interactive_question_loop(
            driver,
            llm
        )

    finally:

        driver.close()

        print("\n✓ Neo4j connection closed.")

    print("\n" + "=" * 70)
    print("PROGRAM FINISHED")
    print("=" * 70)
    print("Thank you for using KG-RAG Assignment.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()
