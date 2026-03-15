import argparse
import sys


def _cmd_serve(args):
    import uvicorn
    from model_router_toolkit.adapters.litellm.app import create_app

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    app = create_app(args.config, models=models)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def _cmd_train(args):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from model_router_toolkit.train import run_train

    kwargs = {}
    if args.mode is not None:
        kwargs["mode"] = args.mode
    if args.device is not None:
        kwargs["device"] = args.device
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.n_seeds is not None:
        kwargs["n_seeds"] = args.n_seeds
    if args.n_keep is not None:
        kwargs["n_keep"] = args.n_keep
    if not args.no_cache and args.prefill_dir:
        kwargs["prefill_dir"] = args.prefill_dir
    if args.prefill_cache is not None:
        kwargs["prefill_cache"] = args.prefill_cache
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.patience is not None:
        kwargs["patience"] = args.patience
    if args.pca_dims is not None:
        kwargs["pca_dims"] = [int(x) for x in args.pca_dims.split(",")]
    if args.models is not None:
        kwargs["models"] = [m.strip() for m in args.models.split(",")]
    if args.features_from is not None:
        kwargs["features_from"] = args.features_from

    run_train(args.config, args.data, args.output_dir, **kwargs)


def _cmd_evaluate(args):
    from model_router_toolkit.evaluate import run_evaluate

    kwargs = {}
    if args.device is not None:
        kwargs["device"] = args.device
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if not args.no_cache and args.prefill_dir:
        kwargs["prefill_dir"] = args.prefill_dir
    if args.prefill_cache is not None:
        kwargs["prefill_cache"] = args.prefill_cache
    if args.models is not None:
        kwargs["models"] = [m.strip() for m in args.models.split(",")]
    if args.features_from is not None:
        kwargs["features_from"] = args.features_from

    run_evaluate(args.config, args.checkpoint, args.data, **kwargs)


def _cmd_collect(args):
    from model_router_toolkit.collect import run_collect

    kwargs = {}
    if args.references is not None:
        kwargs["references_path"] = args.references

    run_collect(args.config, args.questions, args.output, args.judge, **kwargs)


def _cmd_serve_router(args):
    import uvicorn
    from model_router_toolkit.adapters.http.app import create_app

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    app = create_app(args.config, models=models)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def _cmd_proxy(args):
    from model_router_toolkit.adapters.litellm.config_bridge import validate_model_alignment
    from model_router_toolkit.adapters.litellm.proxy import start_proxy

    warnings = validate_model_alignment(args.litellm_config, args.router_config)
    for w in warnings:
        print(f"  Warning: {w}")

    start_proxy(
        args.litellm_config,
        args.router_config,
        host=args.host,
        port=args.port,
    )


def _cmd_proxy_config(args):
    from model_router_toolkit.adapters.litellm.config_bridge import generate_litellm_config

    config = generate_litellm_config(args.config, output=args.output)
    if args.output:
        print(f"  Generated litellm config: {args.output}")
    else:
        import yaml

        print(yaml.dump(config, default_flow_style=False, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(
        prog="model-router",
        description="Model Router Toolkit — intelligent LLM routing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_p = subparsers.add_parser("serve", help="Start the router server")
    serve_p.add_argument("--config", default="configs/prefill-qwen08b.yaml")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--models", default=None, help="Comma-separated model subset to route between (default: all)")
    serve_p.set_defaults(func=_cmd_serve)

    sr_p = subparsers.add_parser(
        "serve-router",
        help="Start a router-only server (routing decisions, no LLM inference)",
    )
    sr_p.add_argument("--config", default="configs/prefill-qwen08b.yaml")
    sr_p.add_argument("--port", type=int, default=8080)
    sr_p.add_argument("--models", default=None, help="Comma-separated model subset to route between (default: all)")
    sr_p.set_defaults(func=_cmd_serve_router)

    train_p = subparsers.add_parser(
        "train", help="Train a routing model from labeled data",
    )
    train_p.add_argument("--config", required=True, help="Pool config YAML")
    train_p.add_argument("--data", required=True, help="Labeled CSV (question, model, isCorrect)")
    train_p.add_argument("--output-dir", default="checkpoints/", help="Output directory")
    train_p.add_argument("--mode", default=None, help="Training mode: auto, single, per_model")
    train_p.add_argument("--device", default=None, help="Device: cpu, cuda, mps, or auto")
    train_p.add_argument("--batch-size", type=int, default=None, help="Extraction batch size")
    train_p.add_argument("--n-seeds", type=int, default=None, help="Ensemble seeds (default: 10)")
    train_p.add_argument("--n-keep", type=int, default=None, help="Ensemble models to keep (default: 5)")
    train_p.add_argument("--prefill-dir", default="cache/", help="Cache dir for prefill features (default: cache/)")
    train_p.add_argument("--no-cache", action="store_true", help="Disable automatic prefill caching")
    train_p.add_argument("--prefill-cache", default=None, help="Pre-extracted PrefillResult .pt file (skips extraction)")
    train_p.add_argument("--epochs", type=int, default=None, help="Max training epochs")
    train_p.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    train_p.add_argument("--pca-dims", default=None, help="PCA dims to sweep, comma-separated (e.g. 50,100,200)")
    train_p.add_argument("--models", default=None, help="Comma-separated model subset to train on (default: all in config)")
    train_p.add_argument("--features-from", default=None, help="Pre-transformed features .pt file (skips extraction + sweep)")
    train_p.set_defaults(func=_cmd_train)

    eval_p = subparsers.add_parser(
        "evaluate", help="Evaluate a trained router checkpoint",
    )
    eval_p.add_argument("--config", required=True, help="Pool config YAML")
    eval_p.add_argument("--checkpoint", required=True, help="Trained checkpoint (.pt)")
    eval_p.add_argument("--data", required=True, help="Test CSV (question, model, isCorrect)")
    eval_p.add_argument("--device", default=None, help="Device: cpu, cuda, mps")
    eval_p.add_argument("--batch-size", type=int, default=None, help="Extraction batch size")
    eval_p.add_argument("--prefill-dir", default="cache/", help="Cache dir for prefill features (default: cache/)")
    eval_p.add_argument("--no-cache", action="store_true", help="Disable automatic prefill caching")
    eval_p.add_argument("--prefill-cache", default=None, help="Pre-extracted PrefillResult .pt file (skips extraction)")
    eval_p.add_argument("--models", default=None, help="Comma-separated model subset to evaluate (default: all)")
    eval_p.add_argument("--features-from", default=None, help="Pre-transformed features .pt file (skips extraction + transforms)")
    eval_p.set_defaults(func=_cmd_evaluate)

    collect_p = subparsers.add_parser(
        "collect", help="Collect training data by running models on questions",
    )
    collect_p.add_argument("--config", required=True, help="Pool config YAML")
    collect_p.add_argument("--questions", required=True, help="Questions file (one per line)")
    collect_p.add_argument("--output", required=True, help="Output CSV path")
    collect_p.add_argument(
        "--judge", default="vote",
        choices=["vote", "llm", "reference"],
        help="Judging method",
    )
    collect_p.add_argument("--references", default=None, help="Reference CSV for reference judging")
    collect_p.set_defaults(func=_cmd_collect)

    proxy_p = subparsers.add_parser(
        "proxy",
        help="Start LiteLLM Proxy with intelligent routing",
    )
    proxy_p.add_argument(
        "--litellm-config", required=True,
        help="LiteLLM proxy config.yaml (model_list, router_settings)",
    )
    proxy_p.add_argument(
        "--router-config", required=True,
        help="Pool config YAML (routing method, checkpoint, models)",
    )
    proxy_p.add_argument("--host", default="0.0.0.0")
    proxy_p.add_argument("--port", type=int, default=4000)
    proxy_p.set_defaults(func=_cmd_proxy)

    pc_p = subparsers.add_parser(
        "proxy-config",
        help="Generate a litellm proxy config.yaml from a pool config",
    )
    pc_p.add_argument("--config", required=True, help="Pool config YAML")
    pc_p.add_argument("--output", default=None, help="Output path (prints to stdout if omitted)")
    pc_p.set_defaults(func=_cmd_proxy_config)

    sc_p = subparsers.add_parser(
        "serve-config", help="Generate serve config from checkpoint (coming soon)",
    )
    sc_p.set_defaults(func=lambda _: print(
        "serve-config is not yet available.\n"
        "Copy and edit one of the example configs in configs/.\n"
        "See configs/prefill-qwen08b.yaml for an example."
    ))

    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n  Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.", file=sys.stderr)
        sys.exit(130)
