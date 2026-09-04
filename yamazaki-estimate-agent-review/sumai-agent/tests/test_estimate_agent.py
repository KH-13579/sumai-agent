"""見積AI（決定論的見積ツール）の単体・APIテスト。"""
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.estimate import EstimateRequest
from app.tools.estimate_calculator import calculate_estimate


def test_standard_wooden_house_is_reproducible():
    request = EstimateRequest(
        floor_area_sqm=99.17355,
        structure="wood",
        floors=2,
        grade="standard",
        equipment=[],
        budget_yen=40_000_000,
    )
    first = calculate_estimate(request)
    second = calculate_estimate(request)

    assert first == second
    assert first.floor_area_tsubo == 30.0
    assert first.expected_yen == 33_000_000
    assert first.low_yen == 29_700_000
    assert first.high_yen == 36_300_000
    assert first.within_budget is True
    assert first.budget_gap_yen == 3_700_000


def test_options_and_budget_overrun_are_explained():
    result = calculate_estimate(EstimateRequest(
        floor_area_sqm=120,
        structure="steel",
        floors=3,
        grade="premium",
        equipment=["solar_4kw", "central_air", "zeh_insulation"],
        budget_yen=40_000_000,
    ))

    assert result.within_budget is False
    assert result.budget_gap_yen < 0
    assert len(result.lines) == 4
    assert len(result.savings_options) >= 3
    assert "正式見積" in result.disclaimer


def test_estimate_api_and_validation():
    client = TestClient(app)
    response = client.post("/api/estimate", json={
        "floor_area_sqm": 100,
        "structure": "wood",
        "floors": 2,
        "grade": "standard",
        "equipment": ["solar_4kw"],
        "budget_yen": 35_000_000,
    })
    assert response.status_code == 200
    assert response.json()["source_url"].startswith("https://www.jhf.go.jp/")

    invalid = client.post("/api/estimate", json={"floor_area_sqm": -1})
    assert invalid.status_code == 422
