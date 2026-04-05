import sqlite3
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_kubernetes_stub() -> None:
    if "kubernetes" in sys.modules:
        return

    kubernetes = types.ModuleType("kubernetes")
    client = types.ModuleType("kubernetes.client")
    config = types.ModuleType("kubernetes.config")

    class ConfigException(Exception):
        pass

    def load_incluster_config() -> None:
        return None

    def load_kube_config() -> None:
        return None

    class ApiException(Exception):
        def __init__(self, status=None, reason=None):
            super().__init__(reason)
            self.status = status

    class ApiClient:
        def call_api(self, *_args, **_kwargs):
            return None, 200, None

    class CoreV1Api:
        def __init__(self):
            self.created = []

        def list_namespace(self):
            return []

        def create_namespaced_pod(self, namespace, pod):
            self.created.append((namespace, pod))

        def read_namespaced_pod(self, _name, _namespace):
            raise NotImplementedError

        def delete_namespaced_pod(self, _name, _namespace):
            return None

        def read_namespaced_pod_log(self, _name, _namespace):
            return ""

    class V1ObjectMeta:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1PodSpec:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1PersistentVolumeClaimVolumeSource:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1EmptyDirVolumeSource:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1Volume:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1VolumeMount:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1Container:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1Pod:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class V1EnvVar:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    config.ConfigException = ConfigException
    config.load_incluster_config = load_incluster_config
    config.load_kube_config = load_kube_config

    client.ApiClient = ApiClient
    client.CoreV1Api = CoreV1Api
    client.V1ObjectMeta = V1ObjectMeta
    client.V1PodSpec = V1PodSpec
    client.V1PersistentVolumeClaimVolumeSource = V1PersistentVolumeClaimVolumeSource
    client.V1EmptyDirVolumeSource = V1EmptyDirVolumeSource
    client.V1Volume = V1Volume
    client.V1VolumeMount = V1VolumeMount
    client.V1Container = V1Container
    client.V1Pod = V1Pod
    client.V1EnvVar = V1EnvVar
    client.exceptions = types.SimpleNamespace(ApiException=ApiException)

    kubernetes.client = client
    kubernetes.config = config

    sys.modules["kubernetes"] = kubernetes
    sys.modules["kubernetes.client"] = client
    sys.modules["kubernetes.config"] = config


def _install_timm_stub() -> None:
    if "timm" in sys.modules:
        return

    timm = types.ModuleType("timm")

    class DummyModel:
        def __init__(self):
            self.loaded_state = None

        def to(self, _device):
            return self

        def parameters(self):
            return []

        def train(self):
            return None

        def __call__(self, _data):
            return object()

        def state_dict(self):
            return {"w": 1}

        def load_state_dict(self, state):
            self.loaded_state = state

    def is_model(name: str) -> bool:
        return not name.startswith("invalid")

    def list_models(pattern: str):
        _ = pattern
        return []

    def create_model(_name: str, pretrained: bool = True):
        _ = pretrained
        return DummyModel()

    timm.is_model = is_model
    timm.list_models = list_models
    timm.create_model = create_model
    sys.modules["timm"] = timm


def _install_torchvision_stub() -> None:
    if "torchvision" in sys.modules:
        return

    torchvision = types.ModuleType("torchvision")
    datasets = types.ModuleType("torchvision.datasets")
    transforms = types.ModuleType("torchvision.transforms")

    class FakeData:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class ToTensor:
        def __call__(self, value):
            return value

    datasets.FakeData = FakeData
    transforms.ToTensor = ToTensor

    torchvision.datasets = datasets
    torchvision.transforms = transforms

    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.datasets"] = datasets
    sys.modules["torchvision.transforms"] = transforms


def _install_groq_stub() -> None:
    if "groq" in sys.modules:
        return

    groq = types.ModuleType("groq")

    class Groq:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

            class _Completions:
                @staticmethod
                def create(*_a, **_k):
                    choice = types.SimpleNamespace(message=types.SimpleNamespace(content="```python\npass\n```"))
                    return types.SimpleNamespace(choices=[choice])

            self.chat = types.SimpleNamespace(completions=_Completions())

    groq.Groq = Groq
    sys.modules["groq"] = groq


_install_kubernetes_stub()
_install_timm_stub()
_install_torchvision_stub()
_install_groq_stub()


@pytest.fixture
def temp_jobs_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "jobs.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'queued',
            config TEXT,
            metrics TEXT DEFAULT '{}',
            retries INTEGER DEFAULT 0,
            pod_name TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def main_module(monkeypatch, temp_jobs_db: Path):
    import main

    def connect_db(_path: str = "jobs.db"):
        conn = sqlite3.connect(temp_jobs_db, check_same_thread=False)
        return conn, conn.cursor()

    monkeypatch.setattr(main, "connect_db", connect_db)
    main.init_db()
    return main


@pytest.fixture
def api_client(main_module):
    return TestClient(main_module.app)


@pytest.fixture
def worker_module(monkeypatch, temp_jobs_db: Path):
    import importlib
    import worker.worker as worker

    worker = importlib.reload(worker)
    monkeypatch.setattr(worker, "DB", str(temp_jobs_db))
    return worker


@pytest.fixture
def controller_module():
    import importlib
    import controller.controller as controller

    return importlib.reload(controller)
