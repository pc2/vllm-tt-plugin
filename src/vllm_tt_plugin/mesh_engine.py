# SPDX-License-Identifier: Apache-2.0
"""Serves one model on a TT mesh that spans the MPI ranks of a tt-run launch.

Every rank owns the devices of its host, and tt-metal needs every rank to issue
the same device operations in the same order.
Rank 0 runs the only engine core with ``TTMeshExecutor``, which sends every
worker call to the other ranks before it runs the call itself.
The other ranks run a ``UniProcExecutor`` and replay the calls in the order
they arrive, so their workers stay in step with the worker of rank 0.
The engine core is a remote engine of an API server started with
``--data-parallel-size-local 0``, as in the headless mode of vLLM.

Usage under tt-run, with the engine arguments of the API server::

    python3 -m vllm_tt_plugin.mesh_engine --model <model> \\
        --data-parallel-address <api host> --data-parallel-rpc-port <port> [args]

``TT_VLLM_MESH_ADDRESS`` is the ``host:port`` on which rank 0 serves the other
ranks, every rank must reach it, it must be an address of the host of rank 0 on
a trusted network, the calls travel as pickled Python objects.
``TT_VLLM_MESH_TIMEOUT`` is the number of seconds rank 0 waits for the other
ranks after it has finished a call itself, 900 by default.

A rank that raises exits with an error, and Open MPI then ends every rank.
A rank whose connection to rank 0 breaks exits with an error as well, so the
other ranks never wait for calls that no longer come.
"""

import os
import pickle
import signal
import time
from concurrent.futures import Future
from typing import Any

import cloudpickle
import zmq
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_tcp_uri
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from zmq.utils.monitor import recv_monitor_message

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

MESH_ADDRESS_ENV = "TT_VLLM_MESH_ADDRESS"
MESH_TIMEOUT_ENV = "TT_VLLM_MESH_TIMEOUT"
_SHUTDOWN_TIMEOUT_S = 120.0
# ZMQ heartbeats detect a peer that is gone even when no call is in flight.
_HEARTBEAT_INTERVAL_MS = 10_000
_HEARTBEAT_TIMEOUT_MS = 60_000


def _mpi_rank() -> int:
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))


def _mpi_size() -> int:
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))


def _mesh_address() -> str:
    address = os.environ.get(MESH_ADDRESS_ENV)
    if not address:
        raise RuntimeError(f"{MESH_ADDRESS_ENV} must be set to the host:port of rank 0")
    return address


def _mesh_timeout() -> float:
    return float(os.environ.get(MESH_TIMEOUT_ENV, "900"))


def _set_heartbeat(socket: zmq.Socket) -> None:
    socket.setsockopt(zmq.HEARTBEAT_IVL, _HEARTBEAT_INTERVAL_MS)
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, _HEARTBEAT_TIMEOUT_MS)
    socket.setsockopt(zmq.HEARTBEAT_TTL, _HEARTBEAT_TIMEOUT_MS)


class _Leader:
    """Sends calls to the other ranks and collects one reply per call and rank."""

    def __init__(self, address: str, followers: int, timeout: float):
        # The messages are pickled, the socket listens only on the address the other ranks use.
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        _set_heartbeat(self._socket)
        self._socket.bind(f"tcp://{address}")
        logger.info("Waiting for %d mesh followers on %s", followers, address)
        identities: set[bytes] = set()
        deadline = time.monotonic() + timeout
        while len(identities) < followers:
            if not self._socket.poll(max(0.0, deadline - time.monotonic()) * 1000):
                raise RuntimeError(
                    f"Only {len(identities)} of {followers} mesh followers connected within {timeout:.0f} s"
                )
            identity, _ = self._socket.recv_multipart()
            identities.add(identity)
        self._identities = sorted(identities)
        self._monitor = self._socket.get_monitor_socket(zmq.EVENT_DISCONNECTED)
        self._pending = 0

    def send(self, message: tuple) -> None:
        data = cloudpickle.dumps(message)
        for identity in self._identities:
            self._socket.send_multipart([identity, data])
        self._pending += 1

    def wait(self, timeout: float, fail_on_disconnect: bool = True) -> None:
        """Collects the replies to every call sent so far and raises on a failed rank."""
        remaining = {identity: self._pending for identity in self._identities}
        errors = []
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        poller.register(self._monitor, zmq.POLLIN)
        deadline = time.monotonic() + timeout
        while any(remaining.values()):
            left = deadline - time.monotonic()
            if left <= 0:
                missing = [i.decode() for i, n in remaining.items() if n]
                raise RuntimeError(f"Mesh ranks {missing} did not answer within {timeout:.0f} s")
            events = dict(poller.poll(left * 1000))
            # Replies are read before disconnects, a rank that answers and then exits is not a failure.
            if self._socket in events:
                identity, data = self._socket.recv_multipart()
                remaining[identity] -= 1
                error = pickle.loads(data)
                if error is not None:
                    errors.append(f"{identity.decode()}: {error}")
            elif self._monitor in events:
                recv_monitor_message(self._monitor)
                if fail_on_disconnect:
                    missing = [i.decode() for i, n in remaining.items() if n]
                    raise RuntimeError(f"A mesh rank disconnected while {missing} had not answered")
        self._pending = 0
        if errors:
            raise RuntimeError("Mesh followers failed: " + "; ".join(errors))

    def close(self) -> None:
        self._socket.disable_monitor()
        self._monitor.close(linger=0)
        self._socket.close(linger=0)
        self._context.term()


class TTMeshExecutor(UniProcExecutor):
    """Runs the worker of rank 0 and mirrors every worker call to the other ranks.

    Every call completes before it returns, the other ranks would not know when
    a deferred output is read, so asynchronous scheduling is not supported.
    After a failed call the ranks may be out of step, every later call fails.
    """

    def _init_executor(self) -> None:
        self._mesh_timeout = _mesh_timeout()
        self._mesh_failed = False
        self._mesh = _Leader(_mesh_address(), _mpi_size() - 1, self._mesh_timeout)
        self._mirrored(lambda: super(TTMeshExecutor, self)._init_executor(), ("init", self.vllm_config))

    def _mirrored(self, local_call, message: tuple) -> Any:
        if self._mesh_failed:
            raise RuntimeError("The mesh ranks are out of step after an earlier failed call")
        self._mesh.send(message)
        try:
            result = local_call()
            self._mesh.wait(self._mesh_timeout)
        except BaseException:
            self._mesh_failed = True
            raise
        return result

    def collective_rpc(  # type: ignore[override]
        self,
        method,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        result = self._mirrored(
            lambda: super(TTMeshExecutor, self).collective_rpc(
                method, timeout, args, kwargs, non_block=False, single_value=single_value
            ),
            ("rpc", method, args, kwargs or {}),
        )
        if not non_block:
            return result
        future: Future = Future()
        future.set_result(result)
        return future

    def shutdown(self) -> None:
        mesh = getattr(self, "_mesh", None)
        if mesh is None:
            super().shutdown()
            return
        self._mesh = None
        mesh.send(("shutdown",))
        try:
            super().shutdown()
            mesh.wait(_SHUTDOWN_TIMEOUT_S, fail_on_disconnect=False)
        finally:
            mesh.close()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return False


def _follow(address: str) -> None:
    """Replays the calls of rank 0 on the worker of this rank until shutdown."""
    # The engine core of rank 0 decides when to stop, a signal to this rank alone would leave the mesh open.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, f"rank{_mpi_rank()}".encode())
    _set_heartbeat(socket)
    monitor = socket.get_monitor_socket(zmq.EVENT_DISCONNECTED)
    socket.connect(f"tcp://{address}")
    socket.send(b"ready")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    poller.register(monitor, zmq.POLLIN)
    executor = None
    while True:
        events = dict(poller.poll())
        if socket not in events:
            recv_monitor_message(monitor)
            raise RuntimeError("The connection to rank 0 broke, the engine core is gone")
        message = cloudpickle.loads(socket.recv())
        kind = message[0]
        try:
            if kind == "init":
                executor = UniProcExecutor(message[1])
            elif kind == "rpc":
                _, method, args, kwargs = message
                executor.collective_rpc(method, args=args, kwargs=kwargs)
            elif kind == "shutdown":
                executor.shutdown()
        except BaseException as exc:
            socket.send(pickle.dumps(repr(exc)))
            raise
        socket.send(pickle.dumps(None))
        if kind == "shutdown":
            break
    socket.disable_monitor()
    monitor.close(linger=0)
    socket.close(linger=1000)
    context.term()


def main() -> None:
    parser = FlexibleArgumentParser(description="TT mesh engine core")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    address = _mesh_address()
    if _mpi_rank() > 0:
        _follow(address)
        return

    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER, headless=True
    )
    # With one data parallel rank and a local engine the parallel config keeps
    # its default address, the API server binds the handshake on the given one.
    if not engine_args.data_parallel_address or not engine_args.data_parallel_rpc_port:
        raise RuntimeError(
            "--data-parallel-address and --data-parallel-rpc-port must match the API server"
        )
    handshake_address = get_tcp_uri(
        engine_args.data_parallel_address, engine_args.data_parallel_rpc_port
    )
    logger.info("Mesh engine core of %d ranks, handshake %s", _mpi_size(), handshake_address)
    EngineCoreProc.run_engine_core(
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=TTMeshExecutor,
        log_stats=not engine_args.disable_log_stats,
        dp_rank=0,
        local_dp_rank=0,
    )


if __name__ == "__main__":
    main()
