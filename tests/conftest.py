from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasonforge.utils import configure_runtime
configure_runtime()


def pytest_addoption(parser):
    parser.addoption("--artifact-run", action="append", default=[], help="Audit a completed real experiment run")


def pytest_generate_tests(metafunc):
    if "artifact_run" in metafunc.fixturenames:
        metafunc.parametrize("artifact_run", [Path(p) for p in metafunc.config.getoption("--artifact-run")],
                             ids=lambda p: p.name)
