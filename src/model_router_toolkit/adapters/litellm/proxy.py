"""Strategy injection hook for the LiteLLM Proxy.

Imports the litellm proxy's FastAPI app, registers a startup event that
patches the proxy's internal Router with our ModelRoutingStrategy, then
runs the server via uvicorn.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_MIN_LITELLM_VERSION = "1.50.0"


def _check_proxy_available() -> None:
    """Verify litellm[proxy] is installed with a compatible version."""
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is not installed. Run: pip install 'model-router-toolkit[proxy]'"
        ) from None

    from packaging.version import Version

    try:
        if Version(litellm.__version__) < Version(_MIN_LITELLM_VERSION):
            logger.warning(
                "litellm %s is older than the tested minimum %s — "
                "proxy integration may not work correctly",
                litellm.__version__,
                _MIN_LITELLM_VERSION,
            )
    except Exception:
        pass

    try:
        from litellm.proxy import proxy_server  # noqa: F401
    except ImportError:
        raise ImportError(
            "litellm proxy extras are not installed. Run: pip install 'litellm[proxy]'"
        ) from None


def _inject_strategy(router_config: str):
    """Patch the litellm proxy's internal Router with our strategy.

    Returns the ModelRoutingStrategy instance so callers can access
    ``last_result`` for response patching.
    """
    import litellm.proxy.proxy_server as proxy_module

    from model_router_toolkit.adapters.litellm.strategy import ModelRoutingStrategy

    llm_router = proxy_module.llm_router
    if llm_router is None:
        raise RuntimeError(
            "litellm proxy did not initialize a Router. "
            "Ensure your litellm config.yaml contains a model_list."
        )

    print("Model Router Toolkit: building strategy from", router_config)
    t0 = time.time()
    strategy = ModelRoutingStrategy.from_config(router_config)
    strategy.set_litellm_router(llm_router)
    llm_router.set_custom_routing_strategy(strategy)
    elapsed = time.time() - t0

    print(f"Model Router Toolkit: strategy registered in {elapsed:.1f}s")
    print(f"  Routing method : {strategy.router.__class__.__name__}")
    print(f"  Tolerance      : {strategy.tolerance}")
    print(f"  Models in proxy: {[d.get('model_name') for d in llm_router.model_list]}")

    try:
        result = strategy.router.route("warmup", tolerance=0.5)
        print(f"  Warmup route   : {result.selected_model}")
    except Exception as e:
        logger.warning("Warmup route failed: %s", e)

    return strategy


def start_proxy(
    litellm_config: str,
    router_config: str,
    *,
    host: str = "0.0.0.0",
    port: int = 4000,
) -> None:
    """Start the litellm proxy with our custom routing strategy injected."""
    _check_proxy_available()

    litellm_config = str(Path(litellm_config).resolve())
    router_config_abs = str(Path(router_config).resolve())

    os.environ["CONFIG_FILE_PATH"] = litellm_config

    from litellm.proxy.proxy_server import app as litellm_app
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    from model_router_toolkit.privacy.middleware import maybe_add_request_redaction_middleware

    _strategy_ref = None

    @litellm_app.on_event("startup")
    async def _inject_routing_strategy():
        nonlocal _strategy_ref
        try:
            _strategy_ref = _inject_strategy(router_config_abs)
        except Exception:
            logger.exception("Failed to inject routing strategy at startup")

    class _RouterProxyMiddleware(BaseHTTPMiddleware):
        """Patch the response ``model`` field to reflect the actual routed
        model (litellm echoes the request model name, not the deployment).
        """

        async def dispatch(self, request: Request, call_next) -> Response:

            response = await call_next(request)

            strategy = _strategy_ref
            if strategy and hasattr(strategy, "last_result") and strategy.last_result:
                selected = strategy.last_result.selected_model
                response.headers["X-Model-Router-Selected"] = selected

                if response.headers.get("content-type", "").startswith("application/json"):
                    body = b""
                    async for chunk in response.body_iterator:
                        body += chunk if isinstance(chunk, bytes) else chunk.encode()

                    import json as _json

                    try:
                        data = _json.loads(body)
                        data["model"] = selected
                        body = _json.dumps(data).encode()
                    except (ValueError, KeyError):
                        pass

                    headers = dict(response.headers)
                    headers["content-length"] = str(len(body))

                    from starlette.responses import Response as StarletteResponse

                    return StarletteResponse(
                        content=body,
                        status_code=response.status_code,
                        headers=headers,
                        media_type=response.media_type,
                    )

            return response

    litellm_app.add_middleware(_RouterProxyMiddleware)
    maybe_add_request_redaction_middleware(litellm_app)

    import uvicorn

    print("\nStarting LiteLLM Proxy with Model Router Toolkit")
    print(f"  LiteLLM config : {litellm_config}")
    print(f"  Router config  : {router_config_abs}")
    print(f"  Listening on   : http://{host}:{port}\n")

    uvicorn.run(litellm_app, host=host, port=port)
