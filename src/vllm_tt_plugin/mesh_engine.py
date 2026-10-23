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
"""

import os
import pickle
import signal
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

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

MESH_ADDRESS_ENV = "TT_VLLM_MESH_ADDRESS"


def _mpi_rank() -> int:
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))


def _mpi_size() -> int:
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))


def _mesh_address() -> str:
    address = os.environ.get(MESH_ADDRESS_ENV)
    if not address:
        raise RuntimeError(f"{MESH_ADDRESS_ENV} must be set to the host:port of rank 0")
    return address


class _Leader:
    """Sends calls to the other ranks and collects their completion."""

    def __init__(self, address: str, followers: int):
        # The messages are pickled, the socket listens only on the address the other ranks use.
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.bind(f"tcp://{address}")
        identities = set()
        logger.info("Waiting for %d mesh followers on %s", followers, address)
        while len(identities) < followers:
            identity, _ = self._socket.recv_multipart()
            identities.add(identity)
        self._identities = sorted(identities)

    def send(self, message: tuple) -> None:
        data = cloudpickle.dumps(message)
        for identity in self._identities:
            self._socket.send_multipart([identity, data])

    def wait(self) -> None:
        errors = []
        for _ in self._identities:
            identity, data = self._socket.recv_multipart()
            error = pickle.loads(data)
            if error is not None:
                errors.append(f"{identity.decode()}: {error}")
        if errors:
            raise RuntimeError("Mesh followers failed: " + "; ".join(errors))

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


class TTMeshExecutor(UniProcExecutor):
    """Runs the worker of rank 0 and mirrors every worker call to the other ranks.

    Every call completes before it returns, the other ranks would not know when
    a deferred output is read, so asynchronous scheduling is not supported.
    """

    def _init_executor(self) -> None:
        self._mesh = _Leader(_mesh_address(), _mpi_size() - 1)
        self._mesh.send(("init", self.vllm_config))
        super()._init_executor()
        self._mesh.wait()

    def collective_rpc(  # type: ignore[override]
        self,
        method,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        self._mesh.send(("rpc", method, args, kwargs or {}))
        result = super().collective_rpc(
            method, timeout, args, kwargs, non_block=False, single_value=single_value
        )
        self._mesh.wait()
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
            mesh.wait()
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
    socket.connect(f"tcp://{address}")
    socket.send(b"ready")
    executor = None
    while True:
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
