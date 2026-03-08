#!/usr/bin/env python3
"""
Standalone encoder server for prefill routing in production.

This script will serve a local encoder model (e.g., Qwen3.5-35B-A3B) over HTTP,
accepting questions and returning prefill hidden states for the Model Router Toolkit's
PrefillRouter. The server runs on GPU and is intended for production deployments where
prefill-based routing is used.
"""


def main() -> None:
    print("Encoder server not yet implemented")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
