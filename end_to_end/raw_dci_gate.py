"""Deprecated in the Paper 3C repo: the gate now lives in `dci_gate.py`
(standalone, routes on the raw DCI). Kept only as a compatibility shim."""
from dci_gate import DCIGate as RawDCIGate  # noqa: F401
