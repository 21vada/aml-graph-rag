# app/retriever.py
"""Retrieve AML evidence subgraphs from Neo4j for LLM explanation."""
import json


class AMLEvidenceRetriever:
    def __init__(self, driver, database="neo4j"):
        self.driver = driver
        self.database = database
        self.cypher = """
        MATCH (s:Account {id: $source_id})-[t:TRANSACTION]->(r:Account {id: $target_id})
        WHERE t.timestamp = $timestamp
        WITH s, r, t
        ORDER BY abs(t.amount_paid - $amount_paid) ASC
        LIMIT 1
        CALL {
            WITH s, r
            MATCH (s)-[rel:TRANSACTION]->(other) WHERE other <> r
            RETURN collect(DISTINCT {
                account: other.id, amount: rel.amount_paid,
                format: rel.payment_format, ts: rel.timestamp
            })[0..8] AS s_sent
        }
        CALL {
            WITH s, r
            MATCH (other)-[rel:TRANSACTION]->(r) WHERE other <> s
            RETURN collect(DISTINCT {
                account: other.id, amount: rel.amount_paid,
                format: rel.payment_format, ts: rel.timestamp
            })[0..8] AS r_received
        }
        RETURN s.id AS source, r.id AS target,
               t.amount_paid AS amount_paid, t.amount_received AS amount_received,
               t.currency_paid AS currency_paid, t.currency_received AS currency_received,
               t.payment_format AS payment_format, t.timestamp AS time, t.label AS label,
               s_sent, r_received
        """

    def search(self, query_text=None, **kw):
        return self.get_search_results(query_text=query_text, **kw)

    def get_search_results(self, query_text=None, **kw):
        p = json.loads(query_text)
        with self.driver.session(database=self.database) as s:
            rec = s.run(self.cypher, **p).single()
        if not rec:
            return {"items": [], "metadata": {"error": "not found"}}
        ctx = (
            f"FLAGGED TRANSACTION\n"
            f"  {rec['source']} -> {rec['target']}\n"
            f"  Amount paid: {rec['amount_paid']} {rec['currency_paid']}\n"
            f"  Amount received: {rec['amount_received']} {rec['currency_received']}\n"
            f"  Format: {rec['payment_format']} | At: {rec['time']} | label={rec['label']}\n"
            f"\nSENDER's recent outgoing:\n"
            f"{json.dumps(rec['s_sent'], indent=2, default=str)}\n"
            f"\nRECEIVER's recent incoming:\n"
            f"{json.dumps(rec['r_received'], indent=2, default=str)}"
        )
        return {
            "items": [{"content": ctx, "metadata": {"source": rec["source"]}}],
            "metadata": {},
        }