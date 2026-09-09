import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------
# STEP 1: Load Environment Variables
# Purpose: Secures sensitive credentials. It reads the .env file and
# injects OPENAI_API_KEY into the system environment so you never 
# hardcode secret keys directly inside your Python code.
# ------------------------------------------------------------------
load_dotenv()


# ------------------------------------------------------------------
# STEP 2: Initialize the LLM (Language Model)
# Purpose: Configures our connection to OpenAI. We use 'gpt-4o-mini'
# to keep token usage low and cheap, and set 'max_tokens' to prevent 
# long, costly responses.
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=100
)


# ------------------------------------------------------------------
# STEP 3: Define System and Human Messages
# Purpose: Directs the AI's persona and structure. 
# - 'system' sets rigid rules, tone, and formatting constraints.
# - 'human' passes the dynamic user inputs ({industry}, {description}).
# ------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a top-tier brand strategist. "
        "Your task is to generate short, memorable product names. "
        "Strict Rule: Always output exactly 3 names, numbered 1 to 3, with no extra intro or outro text."
    ),
    (
        "human",
        "Suggest product names for a {industry} item that does the following: {description}"
    )
])


# ------------------------------------------------------------------
# STEP 4: Initialize Output Parser
# Purpose: Converts the raw AIMessage object returned by OpenAI 
# into a clean, standard Python string that you can easily manipulate.
# ------------------------------------------------------------------
output_parser = StrOutputParser()


# ------------------------------------------------------------------
# STEP 5: Assemble the LCEL Chain (|)
# Purpose: Connects all isolated components into an automated pipeline.
# Data flows sequentially: Prompt Template -> LLM -> Output Parser.
# ------------------------------------------------------------------
chain = prompt | llm | output_parser


# ------------------------------------------------------------------
# STEP 6: Execute the Chain
# Purpose: Triggers the execution. We pass dictionary values for our 
# placeholders, and the chain automatically formats, calls the API, 
# and parses the clean text output.
# ------------------------------------------------------------------
result = chain.invoke({
    "industry": "Electric dron",
    "description": "2 battery based passanger dron that can traver upto 400KM"
})

print("--- AI RESPONSE (System + Human Message) ---")
print(result)