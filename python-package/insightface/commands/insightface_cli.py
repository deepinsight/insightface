#!/usr/bin/env python

from argparse import ArgumentParser

from .model_download import ModelDownloadCommand
from .rec_add_mask_param import RecAddMaskParamCommand

def main():
    parser = ArgumentParser("InsightFace CLI tool", usage="insightface-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="insightface-cli command-line helpers")

    # Register commands
    ModelDownloadCommand.register_subcommand(commands_parser)
    RecAddMaskParamCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()

