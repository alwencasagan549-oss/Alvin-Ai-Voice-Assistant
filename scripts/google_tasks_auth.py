"""Create the OAuth token used by the wake-up briefing to read Google Tasks.

One-time setup:

1. In the Google Cloud console, create (or reuse) a project and enable the
   **Google Tasks API**.
2. Configure the OAuth consent screen (External is fine; add your own Google
   account as a test user).
3. Create an OAuth client of type **Desktop app** and download its JSON to
   ``credentials/client_secret.json``.
4. Run::

       python scripts/google_tasks_auth.py

The script opens a browser for consent and writes the refreshable token to
``credentials/google_tasks_token.json``. Then set in ``.env``::

    BRIEFING_TASKS_ENABLED=true

The token grants read-only access (``tasks.readonly``) and is never used to
modify your task list. ``credentials/`` is gitignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/tasks.readonly"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-secrets",
        default="credentials/client_secret.json",
        help="OAuth client JSON downloaded from the Google Cloud console.",
    )
    parser.add_argument(
        "--token",
        default="credentials/google_tasks_token.json",
        help="Where to write the authorized-user token.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Use the copy/paste console flow instead of a local browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    client_secrets = Path(args.client_secrets)
    token_path = Path(args.token)

    if not client_secrets.is_file():
        print(
            f"error: OAuth client secrets not found at {client_secrets}\n"
            "Download a Desktop-app OAuth client from the Google Cloud console\n"
            "and save it there (or pass --client-secrets).",
            file=sys.stderr,
        )
        return 1

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "error: google-auth-oauthlib is not installed.\n"
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
    if args.console:
        credentials = flow.run_console()
    else:
        credentials = flow.run_local_server(
            port=0,
            authorization_prompt_message="Open this URL to authorize Alvin: {url}",
            success_message="Authorization complete. You can close this window.",
        )

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    print(f"Saved Google Tasks token to {token_path}")
    print("Now set BRIEFING_TASKS_ENABLED=true in .env and restart Alvin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
