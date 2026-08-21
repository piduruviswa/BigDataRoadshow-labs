"""
MEKO Fraud Detection Platform - Synchronous Critical Path
Production Blueprint for Charles Schwab (demo scenario) using MEKO.

Implements the ≤45ms synchronous decisioning path described in
"System Architecture Specification: Autonomous Multi-Agent Fraud
Detection Platform" (Section 5):

  Agent 2 - Behavioral Velocity & Baseline Anomaly Agent
  Agent 3 - Device Reputation & JA4 Telemetry Agent
  Agent 4 - Inline Deterministic Policy Arbiter

No unbounded LLM calls sit on this path - it is deterministic,
rule/heuristic governed, and fully auditable, per Section 6
(Regulatory Compliance & Model Governance).
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="MEKO Fraud Detection Platform",
    description="Synchronous Sub-45ms Real-Time Decisioning & Async Forensic Graph Engine",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stand-in for MEKO fast-state memory (Section 3.2 baseline profiles).
STATE_PROFILES = {
    "ACC-882194": {
        "account_id": "ACC-882194",
        "customer_name": "Eleanor Vance (Private Client)",
        "mean_daily_volume": 420.0,
        "max_historical_tx": 1200.0,
        "authorized_ja4": "t13d1516h2_8daaf6152771_b5e52c8b8791",
        "authorized_asn": "AS7922",
        "known_recipients": ["ZEL-339182", "ZEL-118274", "ACH-SCHWAB-901"],
    },
    "ACC-7701": {
        "account_id": "ACC-7701",
        "customer_name": "John Doe (Synthetic Ring)",
        "mean_daily_volume": 50.0,
        "max_historical_tx": 250.0,
        "authorized_ja4": "t13d1008h1_unknown",
        "authorized_asn": "AS16509",
        "known_recipients": [],
    },
}


class TransactionPayload(BaseModel):
    account_id: str
    amount: float
    recipient_id: str
    channel: str = "P2P_ZELLE"
    client_ip: str
    ja4_signature: str
    device_os: Optional[str] = "Linux x86_64"
    biometric_dwell_ms: Optional[int] = 42


async def agent_behavioral_velocity(payload: TransactionPayload, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Agent 2: sliding-window volume/frequency vs. MEKO account baseline."""
    t0 = time.perf_counter()
    await asyncio.sleep(0.006)

    amount_ratio = payload.amount / max(profile.get("max_historical_tx", 500.0), 1.0)
    is_novel = payload.recipient_id not in profile.get("known_recipients", [])

    risk = min(
        1.0,
        (0.55 if amount_ratio > 1.5 else 0.1)
        + (0.35 if is_novel else 0.0)
        + (0.10 if payload.amount > 2000.0 else 0.0),
    )

    return {
        "risk": round(risk, 3),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "novel_recipient": is_novel,
        "amount_ratio": round(amount_ratio, 2),
    }


async def agent_device_reputation(payload: TransactionPayload, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Agent 3: JA4 TLS fingerprint, ASN/proxy reputation, biometric cadence."""
    t0 = time.perf_counter()
    await asyncio.sleep(0.008)

    is_mismatch = payload.ja4_signature != profile.get("authorized_ja4")
    is_datacenter = "185.220" in payload.client_ip or "198.51" in payload.client_ip
    is_bot = (payload.biometric_dwell_ms or 150) < 60

    risk = min(
        1.0,
        0.05
        + (0.40 if is_mismatch else 0.0)
        + (0.45 if is_datacenter else 0.0)
        + (0.10 if is_bot else 0.0),
    )

    return {
        "risk": round(risk, 3),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "ja4_mismatch": is_mismatch,
        "datacenter_ip": is_datacenter,
        "bot_cadence": is_bot,
    }


@app.post("/api/v1/transactions/evaluate")
async def evaluate_transaction(payload: TransactionPayload):
    """Agent 4: Inline Deterministic Policy Arbiter (SLA budget 5-8ms on top of Agents 2/3)."""
    start = time.perf_counter()

    profile = STATE_PROFILES.get(
        payload.account_id,
        {
            "account_id": payload.account_id,
            "max_historical_tx": 500.0,
            "authorized_ja4": "unknown",
            "known_recipients": [],
        },
    )

    vel_res, dev_res = await asyncio.gather(
        agent_behavioral_velocity(payload, profile),
        agent_device_reputation(payload, profile),
    )

    composite_risk = round((0.45 * vel_res["risk"]) + (0.55 * dev_res["risk"]), 3)
    verdict = "BLOCK" if composite_risk >= 0.75 else "STEP_UP_MFA" if composite_risk >= 0.45 else "APPROVE"

    policies = []
    if dev_res["datacenter_ip"]:
        policies.append("DATACENTER_PROXY_EGRESS")
    if dev_res["ja4_mismatch"]:
        policies.append("UNRECOGNIZED_JA4_FINGERPRINT")
    if vel_res["novel_recipient"]:
        policies.append("NOVEL_BENEFICIARY_BURST")
    if vel_res["amount_ratio"] > 2.0:
        policies.append("DEVIATION_ABOVE_200PCT_MAX")

    return {
        "verdict": verdict,
        "composite_risk_score": composite_risk,
        "pipeline_latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
        "account_id": payload.account_id,
        "amount": payload.amount,
        "agent_telemetry": {
            "velocity_latency_ms": vel_res["latency_ms"],
            "velocity_risk": vel_res["risk"],
            "device_latency_ms": dev_res["latency_ms"],
            "device_risk": dev_res["risk"],
        },
        "triggered_policies": policies,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
