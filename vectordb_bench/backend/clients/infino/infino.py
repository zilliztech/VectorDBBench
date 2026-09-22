import contextlib
import fcntl
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import infino
import numpy as np
import pyarrow as pa

from ..api import VectorDB
from .config import InfinoIndexConfig

log = logging.getLogger(__name__)

_VECTOR_FIELD = "emb"
_ID_FIELD = "id"
_LABEL_FIELD = "label"

# VectorDBBench feeds inserts in small batches (its default is 100 rows), and
# each Infino append() commits a superfile. Committing one superfile per fed
# batch would fragment a large load into thousands of tiny superfiles, making
# both the load and the following optimize pathologically slow. The insert paths
# instead buffer fed rows and commit them as one combined append once this many
# have accumulated; the remainder is flushed when the load's init() scope exits,
# so a corpus smaller than this threshold is still fully persisted. The result is
# a handful of large superfiles regardless of the harness batch size.
_FLUSH_ROWS = 100_000

# Loopback serve mode. With INFINO_BENCH_SERVE=1 the client keeps the embedded
# load path unchanged but serves *search* over a client-server topology: it
# spawns a local `infino-bench-serve` subprocess (bound to 127.0.0.1 only) and
# issues top-k queries over raw TCP. The engine holds one warm index in the
# server process; the benchmark workers are thin TCP clients. The bind address
# is always loopback and chosen here (never configurable to a remote host), so
# the whole benchmark runs on a single machine — the client and the engine can
# never be split across two hosts.
_SERVE_ENV = "INFINO_BENCH_SERVE"
# Seconds to wait for the freshly spawned server to accept connections.
_SERVE_START_TIMEOUT_S = 300.0
# Per-query socket timeout: a warm query is sub-ms, so this only trips if the
# server stalls — letting the worker fail/reconnect instead of wedging forever
# (which would otherwise hang the phase until the harness concurrency timeout).
_SERVE_SOCK_TIMEOUT_S = 120.0


def _serve_enabled() -> bool:
    # Serve mode is the DEFAULT: search runs over a local loopback server so each
    # concurrency worker is a thin TCP client. The embedded path would spawn one
    # engine per worker under the multi-process runner and exhaust memory at
    # higher concurrency. Set INFINO_BENCH_SERVE=0 to force the embedded path.
    return os.environ.get(_SERVE_ENV, "1").strip().lower() not in ("0", "false", "no")


class Infino(VectorDB):
    """VectorDBBench client for Infino, an embedded vector/search engine.

    Infino is in-process: each benchmark worker connects to the same on-disk
    catalog. The instance holds only picklable config so it survives the
    ProcessPoolExecutor(spawn) boundary; the connection and table are opened
    lazily in init().
    """

    # Serialize the load: concurrent writes to a single table are not supported.
    thread_safe: bool = False

    # Take the loader's raw 2-D float32 array instead of a list-of-lists; see
    # _vector_array() for the ~8x memory difference this avoids per shard.
    accepts_ndarray_embeddings: bool = True

    # NonFilter only (base default): Infino's native filtered ANN cannot express
    # the harness's scalar equality / range filters.

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: InfinoIndexConfig,
        collection_name: str = "vdbbench_infino",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "Infino"
        self.dim = dim
        self.data_path = db_config["data_path"]
        # A cache budget without a cache dir is a silent no-op in the
        # engine (no disk cache is created); default the cache next to the
        # catalog so warm queries are actually warm.
        if db_config.get("cache_budget_bytes") and not db_config.get("cache_dir"):
            db_config = {**db_config, "cache_dir": str(Path(self.data_path) / "cache")}
        # Connection tuning (cache budget, cache dir, object-store options); pass only what is set.
        self._connect_opts = {
            k: db_config[k]
            for k in ("cache_budget_bytes", "cache_dir", "storage_options")
            if db_config.get(k) is not None
        }
        self.table_name = collection_name
        self.with_scalar_labels = with_scalar_labels
        self.metric = db_case_config.index_param()["metric"]
        # Vector serving path + serve-time beam, bridged to the engine config
        # before connect (see _apply_search_mode_config).
        self._search_mode = db_case_config.search_mode
        self._ef = db_case_config.ef

        self._conn = None
        self._table = None
        # Rows accumulated by the insert paths as Arrow batches, committed as one
        # large append at _FLUSH_ROWS and when the load's init() scope exits.
        self._buf_batches: list[pa.RecordBatch] = []
        self._buf_rows = 0
        # Loopback serve mode (vector search only): search runs over a local
        # `infino-bench-serve` subprocess instead of the embedded engine. The
        # server projects the dataset `id` column and returns it directly, so
        # the serve path needs no _id map. State below is process-local and is
        # dropped across the spawn boundary (see __getstate__).
        self._serve = _serve_enabled()
        self._server_addr: tuple[str, int] | None = None  # (host, port) once started
        self._server_proc: subprocess.Popen | None = None  # child we spawned (if any)
        self._spawn_lock: threading.Lock | None = None  # guards first-search spawn; set in init()
        self._tls = threading.local()  # per-thread TCP socket to the server
        # Build the schema once so table creation and every append stay in lockstep.
        self._schema = self._build_schema()

        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        if drop_old:
            # The old table's build is gone: retire the serve-ready marker and any
            # server still holding the previous build's snapshot.
            self._serve_ready_path().unlink(missing_ok=True)
            self._kill_recorded_server()
            if self.table_name in conn.list_tables():
                conn.drop_table(self.table_name, purge=True)
        if self.table_name not in conn.list_tables():
            conn.create_table(self.table_name, self._schema, self._index_spec())

    def _apply_search_mode_config(self):
        """Bridge ``search_mode`` and the serve-time beam to the engine's YAML config.

        The engine reads ``vector.search_mode`` and ``vector.hnsw_ef_search`` from
        its config file, loaded once per process from
        ``$XDG_CONFIG_HOME/infino/config.yaml``. We write them here — not through
        ``IndexSpec`` — to keep the engine's public API untouched. The engine
        default is ``ivf`` with the stamped k->ef curve (``hnsw_ef_search = 0``),
        so only values that diverge from that are written; a pure-default run is
        byte-for-byte unchanged. Idempotent, and re-applied in each spawned
        worker before its first connect.
        """
        mode = self._search_mode
        ef = self._ef or 0
        write_mode = bool(mode) and mode != "ivf"
        if not write_mode and ef <= 0:
            return
        lines = ["vector:"]
        if write_mode:
            lines.append(f"  search_mode: {mode}")
        if ef > 0:
            lines.append(f"  hnsw_ef_search: {ef}")
        cfg_root = Path(self.data_path) / "_infino_engine_cfg"
        (cfg_root / "infino").mkdir(parents=True, exist_ok=True)
        (cfg_root / "infino" / "config.yaml").write_text("\n".join(lines) + "\n")
        os.environ["XDG_CONFIG_HOME"] = str(cfg_root)

    def _connect(self):
        self._apply_search_mode_config()
        return infino.connect(self.data_path, **self._connect_opts)

    def __getstate__(self) -> dict:
        # Drop the non-picklable live connection so the instance can cross a
        # process boundary. The buffer is always empty at a process boundary
        # (the load subprocess flushes at init() exit before returning), so it
        # is reset rather than shipped.
        return {
            **self.__dict__,
            "_conn": None,
            "_table": None,
            "_buf_batches": [],
            "_buf_rows": 0,
            # Serve state is process-local: a spawned worker re-elects/rejoins the
            # single local server and opens its own socket (none of these pickle).
            # `_tls` MUST be None here — a threading.local() is unpicklable, so
            # returning one would break the ProcessPoolExecutor(spawn) boundary;
            # _sock() recreates it lazily in the worker.
            "_server_addr": None,
            "_server_proc": None,
            "_spawn_lock": None,
            "_tls": None,
        }

    def _build_schema(self) -> pa.Schema:
        fields = [pa.field(_ID_FIELD, pa.int64(), nullable=False)]
        if self.with_scalar_labels:
            fields.append(pa.field(_LABEL_FIELD, pa.large_utf8(), nullable=False))
        fields.append(pa.field(_VECTOR_FIELD, pa.list_(pa.float32(), self.dim), nullable=False))
        return pa.schema(fields)

    def _index_spec(self) -> infino.IndexSpec:
        return infino.IndexSpec().vector(_VECTOR_FIELD, self.dim, self.metric)

    @contextmanager
    def init(self):
        # Serve-mode SEARCH phase (the build is complete — marker present): stay
        # thin. Do NOT open the embedded table; opening it loads the resident Sq8
        # plane into this process, and one copy per search worker is exactly the
        # per-process cost serve mode exists to avoid. The engine lives in the
        # local server; search goes over TCP (see search_embedding).
        serve_search = self._serve and self._serve_ready_path().exists()
        # init() runs single-threaded before the runner fans out, so this is the
        # safe place to create the lock guarding the first-search spawn. Create it
        # for ANY serve-mode init (not only the marker-present branch): a case that
        # searches before optimize() writes the marker still reaches _ensure_server.
        if self._serve and self._spawn_lock is None:
            self._spawn_lock = threading.Lock()
        if serve_search:
            pass  # thin: no embedded table (see search_embedding)
        elif self._conn is None:
            # Embedded path, or serve-mode LOAD/build phase (append + optimize need
            # the embedded engine). Reuse one connection for the whole process.
            conn = self._connect()
            self._table = conn.open_table(self.table_name)
            self._conn = conn  # assign last so a failed open leaves a clean state to retry
        try:
            yield
        finally:
            # Commit any rows still buffered from the load, in the same
            # (sub)process that inserted them, before it returns — the only point
            # at which a load smaller than _FLUSH_ROWS would otherwise never be
            # persisted. A no-op outside the load path (the buffer is empty).
            if not serve_search:
                self._flush()


    def _vector_array(self, embeddings: Iterable[list[float]] | np.ndarray) -> pa.Array:
        """Build the fixed-size-list vector column for one insert batch.

        A 2-D float32 ``ndarray`` is wrapped around its existing buffer: Arrow
        reads the flat values and the fixed list size supplies the row shape,
        so a shard costs its raw bytes (1M x 768 -> 3.1 GB) and nothing more.
        The list-of-lists the caller would otherwise materialize costs ~32
        bytes per float — a 24-byte Python float object plus an 8-byte
        pointer — i.e. 24.6 GB for that same shard. Several loader workers
        each holding one is what OOM-killed this box (kernel logged 24.9 GB
        of virtual memory per worker against 62 GB of RAM and no swap).

        Anything else (plain lists, generators) takes the general Arrow
        conversion, so non-ndarray callers are unaffected.
        """
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            flat = np.ascontiguousarray(embeddings, dtype=np.float32).reshape(-1)
            return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), self.dim)
        return pa.array(embeddings, type=pa.list_(pa.float32(), self.dim))

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]] | np.ndarray,
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        # Buffer fed rows and commit them as a few large superfiles rather than
        # one per call (see _FLUSH_ROWS); the remainder is flushed at init() exit
        # so every fed row is persisted before search.
        try:
            arrays = [pa.array(metadata, type=pa.int64())]
            if self.with_scalar_labels:
                arrays.append(pa.array(labels_data, type=pa.large_utf8()))
            arrays.append(self._vector_array(embeddings))
            self._buffer(pa.record_batch(arrays, schema=self._schema), len(metadata))
        except Exception as e:
            log.exception("Failed to insert embeddings into Infino")
            return 0, e
        return len(metadata), None

    def _buffer(self, batch: pa.RecordBatch, n_rows: int) -> None:
        """Hold a fed batch, committing the accumulated rows once large enough."""
        self._buf_batches.append(batch)
        self._buf_rows += n_rows
        if self._buf_rows >= _FLUSH_ROWS:
            self._flush()

    def _flush(self) -> None:
        """Commit all buffered rows as a single superfile and clear the buffer.

        Concatenates the buffered batches into one contiguous batch so the load
        commits large superfiles. Inserts are serialized (the client is not
        thread-safe, so the runner drives a single worker), so no lock is needed.
        """
        if not self._buf_batches:
            return
        table = pa.Table.from_batches(self._buf_batches, schema=self._schema).combine_chunks()
        for batch in table.to_batches():
            self._table.append(batch)
        self._buf_batches = []
        self._buf_rows = 0

    # ---- loopback serve mode (INFINO_BENCH_SERVE=1) --------------------------
    #
    # The engine runs in a local `infino-bench-serve` subprocess bound to
    # 127.0.0.1; the benchmark workers are thin TCP clients. Wire (little-endian,
    # one persistent connection, pipelined): request = u32 k, u32 dim, dim*4 f32;
    # response = u32 n, n*8 bytes int64 (the projected dataset id).

    def _server_cmd(self, port: int) -> list[str]:
        cache_bytes = int(self._connect_opts.get("cache_budget_bytes") or 0)
        cmd = [
            sys.executable,
            "-m",
            "infino._bench_serve",
            "--data",
            self.data_path,
            "--table",
            self.table_name,
            "--col",
            _VECTOR_FIELD,
            "--id-col",
            _ID_FIELD,
            "--addr",
            f"127.0.0.1:{port}",
        ]
        if cache_bytes > 0:
            cmd += ["--cache-bytes", str(cache_bytes)]
        return cmd

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def _server_sig(self) -> str:
        """Config signature the running server must match to be reused.

        An ef sweep runs one process per beam over one shared build, each writing a
        different ``vector.hnsw_ef_search``. A server left behind by a crashed beam
        would otherwise be rejoined and silently serve the *previous* beam's ef,
        corrupting the curve. Rejoin only a server whose signature matches; kill and
        respawn otherwise. Includes table + column so a server built for a different
        table (same data_path reused) is never rejoined and asked the wrong vectors.
        """
        return f"{self.table_name}:{_VECTOR_FIELD}:{self._search_mode or 'ivf'}:{self._ef or 0}"

    def _ensure_server(self) -> tuple[str, int]:
        """Start (or rejoin) the single local server and return its loopback address.

        Idempotent across threads (``_spawn_lock``) and processes (an flock over a
        lockfile beside the catalog): the first arrival spawns the child and records
        it; a later arrival rejoins only if the running server's config signature
        matches (see ``_server_sig``), else kills the stale one and respawns. The
        address is always ``127.0.0.1`` — the client never connects the engine to a
        remote host.
        """
        if self._server_addr is not None:
            return self._server_addr
        with self._spawn_lock:
            if self._server_addr is not None:
                return self._server_addr
            # The child must inherit the same engine config (search_mode / ef)
            # the embedded path would have written; this sets XDG_CONFIG_HOME.
            self._apply_search_mode_config()
            lock_path = Path(self.data_path) / ".bench_serve.lock"
            state_path = Path(self.data_path) / ".bench_serve.json"
            with lock_path.open("w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                prev = self._read_server_state(state_path)
                if prev is not None:
                    port, pid, sig = prev
                    if self._port_open("127.0.0.1", port):
                        if sig == self._server_sig():
                            self._server_addr = ("127.0.0.1", port)
                            return self._server_addr
                        # Running, but built for a different beam — retire it.
                        log.info("infino: retiring stale bench server (sig %r != %r)", sig, self._server_sig())
                        self._terminate_pid(pid)
                port = self._free_port()
                log.info("infino: spawning loopback bench server on 127.0.0.1:%d (sig %s)", port, self._server_sig())
                # Redirect the child's stdout/stderr to a file, NOT the inherited
                # ones. Critical when the harness drives the client over an SSH
                # pipe: a detached server that kept the pipe open would hold the
                # remote command open forever (the whole benchmark then looks hung
                # even after it finished). start_new_session detaches it into its
                # own session so it also survives the transient worker that spawns
                # it under the multi-process runner.
                serve_log = (Path(self.data_path) / "bench_serve.log").open("ab")
                proc = subprocess.Popen(
                    self._server_cmd(port),
                    env=os.environ.copy(),
                    stdout=serve_log,
                    stderr=serve_log,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._wait_until_serving("127.0.0.1", port, proc)
                state_path.write_text(json.dumps({"port": port, "pid": proc.pid, "sig": self._server_sig()}))
                self._server_proc = proc
                # Deliberately NO in-process reap (no atexit). Under the harness's
                # multi-process runner each phase (serial, then each concurrency
                # level) runs in a separate short-lived process; if any of them
                # reaped the shared server on exit it would kill it out from under
                # the still-running siblings — that churn stalls workers mid-phase
                # (observed as a collapsed conc-8 QPS) and respawns the server
                # repeatedly. Instead the server is left running (its output goes to
                # a file, so it never holds the caller's pipe) and reaped by the next
                # beam's signature check, the next load's drop_old, or VM teardown.
                self._server_addr = ("127.0.0.1", port)
                return self._server_addr

    def _serve_ready_path(self) -> Path:
        """Marker written after the build+optimize completes; its presence tells
        serve-mode ``init()`` the search phase has begun (stay thin, use the
        server) versus the load phase (open the embedded engine to append)."""
        return Path(self.data_path) / f"{self.table_name}.serveready"

    def _kill_recorded_server(self) -> None:
        """Terminate whatever server the state file records (used at drop_old, when
        the previous build — and any server pinned to its snapshot — is gone)."""
        state = self._read_server_state(Path(self.data_path) / ".bench_serve.json")
        if state is not None:
            self._terminate_pid(state[1])
        (Path(self.data_path) / ".bench_serve.json").unlink(missing_ok=True)

    @staticmethod
    def _read_server_state(path: Path) -> tuple[int, int, str] | None:
        try:
            d = json.loads(path.read_text())
            return int(d["port"]), int(d["pid"]), str(d["sig"])
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _is_our_server(pid: int) -> bool:
        """True only if pid is a live `infino._bench_serve` process. Guards against
        a recorded pid that was reused by an unrelated process (state file persists
        across runs), which matters on a long-lived shared host."""
        try:
            with Path(f"/proc/{pid}/cmdline").open("rb") as fh:
                return b"infino._bench_serve" in fh.read()
        except OSError:
            return False

    def _terminate_pid(self, pid: int) -> None:
        # Only kill a process we can confirm is our bench server (see _is_our_server).
        if not self._is_our_server(pid):
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except OSError:
                return  # already gone
            for _ in range(100):  # up to ~10s to fully exit before we respawn/rebind
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    return  # confirmed dead
            # still alive after the wait → escalate to SIGKILL on the next loop

    def _wait_until_serving(self, host: str, port: int, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + _SERVE_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                msg = f"infino bench server exited early with code {proc.returncode}"
                raise RuntimeError(msg)
            if self._port_open(host, port):
                return
            time.sleep(0.2)
        proc.kill()
        msg = f"infino bench server did not accept connections within {_SERVE_START_TIMEOUT_S:.0f}s"
        raise RuntimeError(msg)

    def _sock(self) -> socket.socket:
        if self._tls is None:  # recreated lazily in a spawned worker (see __getstate__)
            self._tls = threading.local()
        s = getattr(self._tls, "sock", None)
        if s is None:
            host, port = self._server_addr
            s = socket.create_connection((host, port), timeout=_SERVE_SOCK_TIMEOUT_S)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._tls.sock = s
        return s

    def _drop_sock(self) -> None:
        """Discard the cached socket so the next query reconnects. Called on any
        socket error — otherwise a single transient drop leaves a dead socket
        cached and every later query in this worker fails for the whole phase."""
        s = getattr(self._tls, "sock", None) if self._tls is not None else None
        if s is not None:
            with contextlib.suppress(OSError):
                s.close()
            self._tls.sock = None

    @staticmethod
    def _recvall(sock: socket.socket, n: int) -> memoryview:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            r = sock.recv_into(view[got:], n - got)
            if r == 0:
                msg = "infino bench server closed the connection mid-response"
                raise ConnectionError(msg)
            got += r
        return view

    def _search_serve(self, query: list[float], k: int) -> list[int]:
        q = np.ascontiguousarray(query, dtype=np.float32)
        req = struct.pack("<II", k, q.shape[0]) + q.tobytes()
        # One retry on a socket error: drop the (possibly dead) cached socket and
        # reconnect once, so a transient drop doesn't crater this worker.
        for attempt in (0, 1):
            try:
                sock = self._sock()
                sock.sendall(req)
                (n,) = struct.unpack("<I", self._recvall(sock, 4))
                if n == 0:
                    return []
                return np.frombuffer(self._recvall(sock, n * 8), dtype="<i8").tolist()
            except (OSError, ConnectionError, struct.error):
                self._drop_sock()
                if attempt == 1:
                    raise
        return []  # unreachable

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        # Vector search always runs over the local loopback server: the server
        # projects the dataset id, so there is no client-side id map. The server
        # is spawned on the first search, after load and optimize have committed.
        if self._server_addr is None:
            self._ensure_server()
        return self._search_serve(query, k)

    def optimize(self, data_size: int | None = None):
        # Optimize always runs on the embedded engine (it is the build path), even
        # in serve mode. Open the embedded connection directly rather than through
        # init(), whose serve-search branch deliberately leaves it closed.
        if self._conn is None:
            conn = self._connect()
            self._table = conn.open_table(self.table_name)
            self._conn = conn
        self._flush()  # commit any rows not yet persisted (defensive; usually empty)
        self._table.optimize()
        if self._serve:
            # Build complete: from here serve-mode init() stays thin and search
            # goes to the local server (see init() and _serve_ready_path).
            self._serve_ready_path().touch()

    def need_normalize_cosine(self) -> bool:
        return True
