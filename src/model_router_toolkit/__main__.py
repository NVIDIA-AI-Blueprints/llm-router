import argparse


def _cmd_setup(args):
    from model_router_toolkit.setup_wizard import run_setup
    run_setup()


def _cmd_serve(args):
    import uvicorn
    from model_router_toolkit.server.app import create_app

    app = create_app(args.config)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def _cmd_train(args):
    from model_router_toolkit.train import run_train
    run_train(args.config, args.data, args.output_dir)


def _cmd_evaluate(args):
    from model_router_toolkit.evaluate import run_evaluate
    run_evaluate(args.config, args.checkpoint, args.data)


def _cmd_collect(args):
    from model_router_toolkit.collect import run_collect
    run_collect(args.config, args.questions, args.output, args.judge)


def _cmd_serve_config(args):
    print("Not yet implemented")


def main():
    parser = argparse.ArgumentParser(
        prog="model-router",
        description="Model Router Toolkit — intelligent LLM routing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_p = subparsers.add_parser("setup", help="Interactive setup wizard")
    setup_p.set_defaults(func=_cmd_setup)

    serve_p = subparsers.add_parser("serve", help="Start the router server")
    serve_p.add_argument("--config", default="configs/generated.yaml")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.set_defaults(func=_cmd_serve)

    train_p = subparsers.add_parser("train", help="Train a routing model")
    train_p.add_argument("--config", required=True)
    train_p.add_argument("--data", required=True)
    train_p.add_argument("--output-dir", default="checkpoints/")
    train_p.set_defaults(func=_cmd_train)

    eval_p = subparsers.add_parser("evaluate", help="Evaluate a trained router")
    eval_p.add_argument("--config", required=True)
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--data", required=True)
    eval_p.set_defaults(func=_cmd_evaluate)

    collect_p = subparsers.add_parser("collect", help="Collect training data")
    collect_p.add_argument("--config", required=True)
    collect_p.add_argument("--questions", required=True)
    collect_p.add_argument("--output", required=True)
    collect_p.add_argument("--judge", default="vote", choices=["vote", "llm", "reference"])
    collect_p.set_defaults(func=_cmd_collect)

    sc_p = subparsers.add_parser("serve-config", help="Generate serve config from checkpoint")
    sc_p.set_defaults(func=_cmd_serve_config)

    args = parser.parse_args()
    args.func(args)
