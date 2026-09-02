from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.main import app


def test_login_with_correct_password_sets_a_session_cookie_that_grants_access():
    client = TestClient(app)
    password = get_settings().admin_password

    login_response = client.post("/admin/login", json={"password": password})

    assert login_response.status_code == 200
    assert "mse_admin_session" in login_response.cookies

    files_response = client.get("/admin/files")
    assert files_response.status_code == 200


def test_login_with_wrong_password_is_rejected():
    client = TestClient(app)

    response = client.post("/admin/login", json={"password": "definitely-wrong"})

    assert response.status_code == 401
    assert "mse_admin_session" not in response.cookies


def test_admin_route_without_a_session_is_rejected():
    client = TestClient(app)

    response = client.get("/admin/files")

    assert response.status_code == 401


def test_admin_route_with_a_garbage_cookie_is_rejected():
    client = TestClient(app)
    client.cookies.set("mse_admin_session", "not-a-real-token")

    response = client.get("/admin/files")

    assert response.status_code == 401


def test_logout_clears_the_session_cookie():
    client = TestClient(app)
    password = get_settings().admin_password
    client.post("/admin/login", json={"password": password})

    logout_response = client.post("/admin/logout")

    assert logout_response.status_code == 200
    files_response = client.get("/admin/files")
    assert files_response.status_code == 401
