"""Send project notifications through an SMTP account from environment variables."""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


ENV_NAMES = {
    "host": "INF_MASKING_MAIL_SMTP_HOST",
    "port": "INF_MASKING_MAIL_SMTP_PORT",
    "sender": "INF_MASKING_MAIL_SENDER",
    "password": "INF_MASKING_MAIL_PASSWORD",
    "recipient": "INF_MASKING_MAIL_RECIPIENT",
}


def load_config() -> dict[str, str | int]:
    values: dict[str, str | int] = {}
    missing: list[str] = []
    for key, env_name in ENV_NAMES.items():
        value = os.environ.get(env_name, "").strip()
        if not value:
            missing.append(env_name)
        values[key] = value
    if missing:
        raise RuntimeError("missing mail environment variables: " + ", ".join(missing))
    try:
        values["port"] = int(str(values["port"]))
    except ValueError as exc:
        raise RuntimeError("INF_MASKING_MAIL_SMTP_PORT must be an integer") from exc
    return values


def read_body(args: argparse.Namespace) -> str:
    if args.body_file:
        return Path(args.body_file).read_text(encoding="utf-8")
    if args.body is not None:
        return args.body
    raise RuntimeError("one of --body-file or --body is required")


def send_email(subject: str, body: str) -> None:
    config = load_config()
    recipients = [
        item.strip()
        for item in str(config["recipient"]).replace(";", ",").split(",")
        if item.strip()
    ]
    if not recipients:
        raise RuntimeError("INF_MASKING_MAIL_RECIPIENT is empty")

    message = EmailMessage()
    message["From"] = str(config["sender"])
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(
        str(config["host"]), int(str(config["port"])), timeout=30, context=context
    ) as server:
        server.login(str(config["sender"]), str(config["password"]))
        server.send_message(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file")
    parser.add_argument("--body")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate environment variables without sending an email",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        load_config()
        if args.check_config:
            print("MAIL_CONFIG_OK")
            return 0
        send_email(args.subject, read_body(args))
        print("EMAIL_SENT")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must return a useful status.
        print(f"EMAIL_SEND_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
