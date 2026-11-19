"""Headless smoke test: every dashboard page renders without an exception."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "dashboard", "app.py")


def test_every_page_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    failures = {}
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        if at.exception:
            failures[page] = [str(e.value)[:300] for e in at.exception]
    assert not failures, failures
