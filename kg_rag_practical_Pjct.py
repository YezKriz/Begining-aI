# ============================================================
# KG-RAG WITH NEO4J AURA DB
# SINGLE FILE LEARNING PROJECT
# ============================================================
#
# STAGE 1:
# Neo4j AuraDB connection test
#
# Goal:
# Check whether Python can securely connect to our
# Neo4j AuraDB database.
#
# We are using the certificate-handling idea from the
# working Neo4j example.
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import os
import sys
import certifi

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ------------------------------------------------------------
# STEP 2: CONFIGURE SSL CERTIFICATE HANDLING
# ------------------------------------------------------------
#
# Why are we doing this?
#
# Our previous Neo4j attempt had an SSL certificate
# verification problem.
#
# certifi provides a trusted CA certificate bundle.
#
# On Windows, we tell Python where this certificate bundle
# is located.
# ------------------------------------------------------------

if sys.platform == "win32":
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

print("=" * 60)
print("KG-RAG WITH NEO4J AURA DB")
print("STAGE 1 - NEO4J CONNECTION TEST")
print("=" * 60)

print("\n[1] SSL certificate configuration")
print("    certifi CA bundle:")
print(f"    {certifi.where()}")


# ------------------------------------------------------------
# STEP 3: LOAD .ENV FILE
# ------------------------------------------------------------
#
# Our Neo4j username and password should NOT be written
# directly inside this Python file.
#
# They remain inside the .env file.
# ------------------------------------------------------------

print("\n[2] Loading environment variables...")

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ------------------------------------------------------------
# STEP 4: CHECK THAT REQUIRED VALUES EXIST
# ------------------------------------------------------------

print("\n[3] Checking Neo4j configuration...")

if not NEO4J_URI:
    raise ValueError("NEO4J_URI is missing from .env")

if not NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME is missing from .env")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is missing from .env")


print("    NEO4J_URI      :", NEO4J_URI)
print("    NEO4J_USERNAME :", NEO4J_USERNAME)
print("    Password loaded: YES")


# ------------------------------------------------------------
# STEP 5: CREATE NEO4J DRIVER
# ------------------------------------------------------------
#
# The Driver is responsible for communication between
# our Python program and Neo4j AuraDB.
#
# We are intentionally using the URI from .env first.
#
# We will NOT change the URI to bolt+s:// yet.
# First we want to see what happens with the properly
# configured certificate environment.
# ------------------------------------------------------------

print("\n[4] Creating Neo4j driver...")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

print("    Neo4j driver created")


# ------------------------------------------------------------
# STEP 6: VERIFY CONNECTIVITY
# ------------------------------------------------------------
#
# verify_connectivity() checks whether the driver can
# actually communicate with Neo4j.
#
# This is better than simply creating the driver because
# creating a driver does not necessarily mean that the
# database connection has succeeded.
# ------------------------------------------------------------

print("\n[5] Testing Neo4j connectivity...")

try:

    driver.verify_connectivity()

    print("    ✅ Neo4j connectivity successful!")

except Exception as e:

    print("    ❌ Neo4j connectivity failed!")
    print()
    print("    Error type :", type(e).__name__)
    print("    Error      :", e)

    driver.close()

    raise SystemExit(1)


# ------------------------------------------------------------
# STEP 7: RUN A SIMPLE CYPHER QUERY
# ------------------------------------------------------------
#
# Now that connectivity works, let's actually send a
# Cypher query to Neo4j.
#
# RETURN 1 AS test
#
# This does not modify the database.
# It simply proves that Neo4j can execute a query.
# ------------------------------------------------------------

print("\n[6] Running test Cypher query...")

try:

    with driver.session() as session:

        result = session.run(
            "RETURN 1 AS test"
        )

        record = result.single()

        print("    Query result:", record["test"])

        print("    ✅ Cypher query successful!")

except Exception as e:

    print("    ❌ Cypher query failed!")
    print()
    print("    Error type :", type(e).__name__)
    print("    Error      :", e)

    driver.close()

    raise SystemExit(1)


# ------------------------------------------------------------
# STEP 8: CLOSE THE DRIVER
# ------------------------------------------------------------

print("\n[7] Closing Neo4j connection...")

driver.close()

print("    Connection closed.")


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("✅ STAGE 1 COMPLETE")
print("=" * 60)

print()
print("Python successfully connected to Neo4j AuraDB")
print("and executed a Cypher query.")
print()
print("Next stage will add our KG-RAG entities.")
print("=" * 60)