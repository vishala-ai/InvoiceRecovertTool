"""Authenticate integrated UI tests with isolated, genuine local accounts."""
from pathlib import Path

from recovery.local_auth import authenticate, bootstrap_owner, initialize


TEST_USERNAME = 'workspace-test-owner'
TEST_PASSWORD = 'Test-only workspace password 29!'


def provision_account(database):
    """Create an account only in the caller's temporary authentication database."""
    initialize(database)
    bootstrap_owner(database, TEST_USERNAME, 'Synthetic test reviewer', TEST_PASSWORD)


def sign_in_app(app, database):
    """Supply a real server-validated session, without bypassing the app gate."""
    app.session_state['auth:token'] = authenticate(database, TEST_USERNAME, TEST_PASSWORD)
    app.session_state['auth:database'] = str(Path(database).resolve())
    return app
