"""
Automated Test Suite (Section 7 of the architecture spec).

Validates that the synchronous critical path stays within the ≤45ms SLA
and produces the correct verdict for both a legitimate private-client
transaction and a hostile datacenter-proxy account-takeover burst.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from main import app


@pytest.mark.asyncio
async def test_legitimate_client_under_sla():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post(
            "/api/v1/transactions/evaluate",
            json={
                "account_id": "ACC-882194",
                "amount": 150.00,
                "recipient_id": "ZEL-339182",
                "client_ip": "73.189.44.12",
                "ja4_signature": "t13d1516h2_8daaf6152771_b5e52c8b8791",
            },
        )
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "APPROVE"
    assert data["composite_risk_score"] < 0.30
    assert data["pipeline_latency_ms"] <= 45.0


@pytest.mark.asyncio
async def test_velocity_ato_burst_hard_block():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post(
            "/api/v1/transactions/evaluate",
            json={
                "account_id": "ACC-882194",
                "amount": 2500.00,
                "recipient_id": "ZEL-NEW-99120",
                "client_ip": "185.220.101.5",
                "ja4_signature": "t13d0305h1_datacenter_nordvpn",
            },
        )
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] == "BLOCK"
    assert data["composite_risk_score"] >= 0.75
    assert "DATACENTER_PROXY_EGRESS" in data["triggered_policies"]
    assert data["pipeline_latency_ms"] <= 45.0
