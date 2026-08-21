"""
MEKO Demo Server - wraps the spec's synchronous critical path (main.py,
unmodified) with the asynchronous deep-forensic path described in the
architecture spec but not given as code there:

  Agent 5 - Deep Graph Traversal & Mule Ring Agent      (SLA 500ms-3000ms)
  Agent 6 - Forensic Synthesis & SAR Generation Agent    (SLA 2s-15s)

Also serves the "Schwab Unified Fraud & SecOps Console" static frontend
and a handful of canned scenarios so the whole story - sub-45ms inline
decisioning, then multi-hop entity resolution, then an auto-drafted
FinCEN SAR narrative - can be driven live from one browser tab.

Run with:  uvicorn demo_app:app --reload --port 8000
"""

import asyncio
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from main import app  # the exact synchronous /api/v1/transactions/evaluate path

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

with open(BASE_DIR / "seed_graph.json") as f:
    GRAPH = json.load(f)

with open(BASE_DIR / "seed_profiles.json") as f:
    PROFILES = json.load(f)

NODES_BY_ID = {n["id"]: n for n in GRAPH["nodes"]}

ADJACENCY: Dict[str, List[dict]] = defaultdict(list)
for edge in GRAPH["edges"]:
    ADJACENCY[edge["source"]].append({**edge, "_dir": "out"})
    ADJACENCY[edge["target"]].append({**edge, "_dir": "in"})

MAX_HOPS = 3

# In-memory case log so the console can show an audit trail (MEKO Audit Vault, Section 6).
CASE_LOG: List[dict] = []


def _bfs_ring(start_id: str, max_hops: int = MAX_HOPS):
    """
    Agent 5: 2-3 hop entity resolution over the MEKO graph datapack.
    Community detection is approximated here as the connected component
    reachable from `start_id` within `max_hops` - sufficient to
    demonstrate the pattern on a small seed graph; production MEKO
    deployments run true Louvain community detection over the full
    entity graph.
    """
    if start_id not in NODES_BY_ID:
        return None

    visited: Set[str] = {start_id}
    frontier = deque([(start_id, 0)])
    touched_edges = []

    while frontier:
        node_id, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for edge in ADJACENCY.get(node_id, []):
            other = edge["target"] if edge["_dir"] == "out" else edge["source"]
            touched_edges.append(edge)
            if other not in visited:
                visited.add(other)
                frontier.append((other, depth + 1))

    ring_accounts = sorted(
        nid for nid in visited if NODES_BY_ID[nid]["type"] == "Account" and nid != start_id
    )
    ring_accounts = [start_id] + ring_accounts if NODES_BY_ID[start_id]["type"] == "Account" else ring_accounts

    shared_artifacts = [
        NODES_BY_ID[nid]
        for nid in visited
        if NODES_BY_ID[nid]["type"] in ("DeviceHash", "PhoneNumber", "Address")
    ]

    mule_nodes = [NODES_BY_ID[nid] for nid in visited if NODES_BY_ID[nid]["type"] == "ExternalMule"]

    # Edges are indexed under both endpoints for undirected BFS, so a single
    # edge can show up twice in touched_edges (once from each side) - dedupe
    # by identity before using it for the subgraph or summing amounts.
    dedup_edges = {(e["source"], e["target"], e["relation"]): e for e in touched_edges}

    pending_exposure = sum(
        e.get("amount", 0.0)
        for e in dedup_edges.values()
        if e["relation"] == "PENDING_OUTBOUND_ACH" and e["source"] in ring_accounts
    )

    subgraph = {
        "nodes": [NODES_BY_ID[nid] for nid in visited],
        "edges": list(dedup_edges.values()),
    }

    return {
        "ring_members": ring_accounts,
        "shared_artifacts": shared_artifacts,
        "mule_beneficiaries": mule_nodes,
        "pending_exposure_usd": round(pending_exposure, 2),
        "subgraph": subgraph,
    }


@app.get("/api/v1/graph/investigate/{account_id}")
async def investigate_graph(account_id: str):
    """Agent 5: Deep Graph Traversal & Mule Ring Agent (async, off critical path)."""
    start = time.perf_counter()
    await asyncio.sleep(0.9)  # representative of a real 2-3 hop graph datapack query

    result = _bfs_ring(account_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No graph presence for account {account_id}")

    ring_detected = len(result["ring_members"]) >= 2 and len(result["mule_beneficiaries"]) > 0

    payload = {
        "account_id": account_id,
        "ring_detected": ring_detected,
        "ring_size": len(result["ring_members"]),
        "ring_members": result["ring_members"],
        "shared_artifacts": result["shared_artifacts"],
        "mule_beneficiaries": result["mule_beneficiaries"],
        "pending_exposure_usd": result["pending_exposure_usd"],
        "subgraph": result["subgraph"],
        "method": "3-hop BFS traversal + connected-component clustering (Louvain in production MEKO)",
        "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
    }

    CASE_LOG.append({
        "type": "GRAPH_INVESTIGATION",
        "account_id": account_id,
        "ring_detected": ring_detected,
        "at": datetime.now(timezone.utc).isoformat(),
    })

    return payload


def _sar_narrative(account_id: str, investigation: dict) -> str:
    members = ", ".join(investigation["ring_members"])
    artifacts = "; ".join(a["label"] for a in investigation["shared_artifacts"])
    mule = investigation["mule_beneficiaries"][0]["label"] if investigation["mule_beneficiaries"] else "an external beneficiary"
    exposure = investigation["pending_exposure_usd"]
    filed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"FinCEN FORM 111 - SUSPICIOUS ACTIVITY REPORT (DRAFT)\n"
        f"Filing generated by MEKO Forensic Synthesis Agent | {filed_at}\n\n"
        f"Subject Accounts: {members}\n\n"
        f"Narrative:\n"
        f"MEKO's deep graph agent identified {len(investigation['ring_members'])} accounts "
        f"({members}) sharing common digital artifacts inconsistent with unrelated customers: "
        f"{artifacts}. Within a 3-hop traversal of the entity graph, each subject account "
        f"initiated a pending outbound ACH transfer of between $4,890.00 and $4,950.00 - each "
        f"individually below the $5,000 structuring-review threshold - to a common external "
        f"beneficiary ({mule}). Aggregate pending exposure across the ring is "
        f"${exposure:,.2f}. Transaction structuring below reporting thresholds combined with "
        f"shared device fingerprint, recovery phone number, and CMRA billing address is "
        f"consistent with a synthetic-identity mule network established to fragment and "
        f"launder funds via a single offshore beneficiary account.\n\n"
        f"Recommended Action: Freeze pending disbursements on subject accounts pending manual "
        f"review; escalate to Schwab Financial Crimes Unit for Form 111 filing within the "
        f"regulatory 30-day window.\n\n"
        f"Generated by: MEKO Agent 6 (Forensic Synthesis & SAR Generation) | "
        f"Audit Ref: SAR-{account_id}-{int(time.time())}"
    )


@app.post("/api/v1/sar/generate/{account_id}")
async def generate_sar(account_id: str):
    """Agent 6: Forensic Synthesis & SAR Generation Agent (async, 2-15s SLA)."""
    start = time.perf_counter()

    investigation = _bfs_ring(account_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail=f"No graph presence for account {account_id}")

    ring_detected = len(investigation["ring_members"]) >= 2 and len(investigation["mule_beneficiaries"]) > 0
    if not ring_detected:
        raise HTTPException(status_code=422, detail="No syndicate pattern detected - SAR not warranted")

    await asyncio.sleep(2.2)  # representative of narrative synthesis + evidence graph compilation

    narrative = _sar_narrative(account_id, investigation)

    CASE_LOG.append({
        "type": "SAR_GENERATED",
        "account_id": account_id,
        "ring_members": investigation["ring_members"],
        "at": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "form_type": "FinCEN Form 111",
        "case_id": f"SAR-{account_id}-{int(time.time())}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ring_members": investigation["ring_members"],
        "pending_exposure_usd": investigation["pending_exposure_usd"],
        "narrative": narrative,
        "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
    }


@app.get("/api/v1/audit/log")
async def audit_log():
    """MEKO Audit Vault (Section 6): immutable record of agent decisions for this session."""
    return {"entries": CASE_LOG}


@app.get("/api/v1/demo/scenarios")
async def demo_scenarios():
    """Canned inputs for the console's one-click scenario buttons."""
    return {
        "scenarios": [
            {
                "id": "legit_purchase",
                "label": "Legit Private-Client Transfer",
                "description": "Eleanor Vance sends $150 to a known Zelle recipient from her usual device.",
                "payload": {
                    "account_id": "ACC-882194",
                    "amount": 150.00,
                    "recipient_id": "ZEL-339182",
                    "client_ip": "73.189.44.12",
                    "ja4_signature": "t13d1516h2_8daaf6152771_b5e52c8b8791",
                    "biometric_dwell_ms": 210,
                },
            },
            {
                "id": "ato_burst",
                "label": "Sub-Second ATO Burst Attack",
                "description": "Same account, $2,500 to a brand-new recipient from a datacenter proxy with a spoofed JA4.",
                "payload": {
                    "account_id": "ACC-882194",
                    "amount": 2500.00,
                    "recipient_id": "ZEL-NEW-99120",
                    "client_ip": "185.220.101.5",
                    "ja4_signature": "t13d0305h1_datacenter_nordvpn",
                    "biometric_dwell_ms": 18,
                },
            },
            {
                "id": "sleeper_mule",
                "label": "Sleeper Mule Ring Transfer",
                "description": "A synthetic-identity account moves $4,900 - just under the $5k threshold - to an offshore beneficiary shared with two other linked accounts.",
                "payload": {
                    "account_id": "ACC-7701",
                    "amount": 4900.00,
                    "recipient_id": "MULE-BENEFICIARY-EXT",
                    "client_ip": "24.18.201.9",
                    "ja4_signature": "t13d1008h1_unknown",
                    "biometric_dwell_ms": 95,
                },
                "graph_followup_account": "ACC-7701",
            },
        ],
        "profiles": PROFILES,
    }


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
