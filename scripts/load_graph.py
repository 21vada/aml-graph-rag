# scripts/load_graph.py
"""Load the sample graph CSV into Neo4j.

Run once after `docker compose up`:
    docker compose exec api python scripts/load_graph.py
"""
import csv
import os
from neo4j import GraphDatabase

URI      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "amlpassword123")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_graph.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

print(f"Connecting to {URI} ...")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD), connection_acquisition_timeout=30)
driver.verify_connectivity()

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))
print(f"Loaded {len(rows):,} rows from CSV")

with driver.session() as s:
    print("Clearing existing graph ...")
    s.run("MATCH (n) DETACH DELETE n")
    print("Creating constraint ...")
    s.run("CREATE CONSTRAINT account_id IF NOT EXISTS "
          "FOR (a:Account) REQUIRE a.id IS UNIQUE")

BATCH = 500
with driver.session() as s:
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i+BATCH]
        s.run("""
            UNWIND $rows AS row
            MERGE (a:Account {id: row.source})
            MERGE (b:Account {id: row.target})
            CREATE (a)-[:TRANSACTION {
                amount_paid:       toFloat(row.amount_paid),
                amount_received:   toFloat(row.amount_received),
                currency_paid:     row.currency_paid,
                currency_received: row.currency_received,
                payment_format:    row.payment_format,
                timestamp:         row.timestamp,
                label:             toInteger(row.label)
            }]->(b)
        """, rows=batch)
        print(f"  loaded {min(i+BATCH, len(rows)):,}/{len(rows):,}")

with driver.session() as s:
    n = s.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
    t = s.run("MATCH ()-[r:TRANSACTION]->() RETURN count(r) AS c").single()["c"]
    l = s.run("MATCH ()-[r:TRANSACTION]->() WHERE r.label=1 "
              "RETURN count(r) AS c").single()["c"]
print(f"✅ Loaded {n:,} accounts | {t:,} transactions | {l:,} laundering")
driver.close()