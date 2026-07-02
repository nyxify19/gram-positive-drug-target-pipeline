"""Shared infrastructure: HTTP, SQLite, logging, tool detection."""
from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional, Sequence

import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore[no-redef]

from pipeline.config import LOGGER

_THREAD_LOCAL = threading.local()


def configure_logging() -> None:
    """Configure root logging once, with a compact timestamped format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    plt.rcParams["font.size"] = 12


def make_http_session() -> requests.Session:
    """Build an HTTP session that retries with exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {"User-Agent": "drugtarget-pipeline/2.0 (research)"}
    )
    session.trust_env = True
    return session


def http() -> requests.Session:
    """Return a per-thread HTTP session."""
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = make_http_session()
        _THREAD_LOCAL.session = session
    return session


def find_tool(*names: str) -> Optional[str]:
    """Return the path to the first executable found on PATH, or None."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


@contextmanager
def db_connection(path: str, timeout: float = 30.0) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection, committing on success and always closing."""
    conn = sqlite3.connect(path, timeout=timeout)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def cached_parallel_fetch(
    accessions: Sequence[str],
    cache_db: str,
    table: str,
    fetch_fn: Callable[[str], tuple],
    *,
    workers: int,
    should_cache: Callable[[tuple], bool] = lambda _row: True,
    progress_every: int = 1000,
    label: str = "fetch",
) -> dict[str, int]:
    """Fetch per-accession results concurrently with a SQLite cache."""
    status_counts: dict[str, int] = {}
    if not accessions:
        return status_counts

    lock = threading.Lock()
    completed = 0
    total = len(accessions)

    def persist(result: tuple) -> None:
        nonlocal completed
        status = result[-1]
        with lock:
            status_counts[status] = status_counts.get(status, 0) + 1
            if should_cache(result):
                row = (*result[:-1], datetime.now(timezone.utc).isoformat())
                placeholders = ",".join("?" * len(row))
                with db_connection(cache_db) as conn:
                    conn.execute(
                        f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})",
                        row,
                    )
            completed += 1
            if completed % progress_every == 0:
                LOGGER.info("  [%s] %d / %d", label, completed, total)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_acc = {pool.submit(fetch_fn, acc): acc for acc in accessions}
        for future in as_completed(future_to_acc):
            try:
                persist(future.result())
            except Exception as exc:  # noqa: BLE001 - keep the pool alive
                with lock:
                    status_counts["error"] = status_counts.get("error", 0) + 1
                LOGGER.warning(
                    "[%s] %s failed: %s", label, future_to_acc[future], exc
                )
    return status_counts
