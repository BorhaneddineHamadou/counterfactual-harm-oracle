"""Scenario package of the openpilot simulation harness.

Only `base.py` (the SceRTScenario interface and road-layout helpers) is
needed by the counterfactual-harm templates in `../templates.py`. The
harness's own scenario catalogue is not part of this package, so the
registry below is empty; the bridge resolves the five templates from
`templates.PROXIMA_SCENARIOS` instead.
"""
ALL_SCENARIOS = []
