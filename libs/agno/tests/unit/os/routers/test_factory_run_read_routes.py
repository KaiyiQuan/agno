"""AgentOS factory-backed run read routes must not require request context
to be omitted (issue #10890): GET run/list/checkpoint routes pass
request=request to resolve_agent/resolve_team so factory resources resolve."""

import pytest
from fastapi.testclient import TestClient

from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS
from agno.team import Team, TeamFactory


@pytest.fixture
def client():
    factory = TeamFactory(
        id="my-team",
        db=InMemoryDb(),
        factory=lambda ctx: Team(
            id="my-team", model=OpenAIResponses(id="gpt-5-mini"), members=[]
        ),
    )
    app = AgentOS(teams=[factory], db=InMemoryDb()).get_app()
    with TestClient(app) as client:
        yield client


def test_factory_team_run_read_routes_resolve(client):
    # POST creates the run even if the model is not reachable (no API key).
    resp = client.post("/teams/my-team/runs", data={"message": "hi", "stream": "false", "session_id": "s1"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    # Before the fix these returned 400 "Request context is required for factory teams".
    assert client.get(f"/teams/my-team/runs/{run_id}", params={"session_id": "s1"}).status_code == 200
    assert client.get("/teams/my-team/runs", params={"session_id": "s1"}).status_code == 200
    assert client.get(f"/teams/my-team/runs/{run_id}/checkpoints", params={"session_id": "s1"}).status_code == 200
    assert (
        client.get(f"/teams/my-team/runs/{run_id}/checkpoints/0", params={"session_id": "s1"}).status_code
        in (200, 404)
    )
