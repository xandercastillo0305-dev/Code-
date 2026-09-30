"""Start the web app and open the login page in the default browser.

    python run.py                 # http://127.0.0.1:5000/login opens automatically
    python run.py --port 8080
    python run.py --no-browser

Same app as `flask --app app run`; this only adds the automatic browser opening.
"""
import argparse
import threading
import webbrowser

from app import create_app


def main():
    ap = argparse.ArgumentParser(description="Run the Deepfake HITL prototype.")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/login"
    if not args.no_browser:
        # Open shortly after start-up, once the server is listening.
        threading.Timer(1.5, webbrowser.open, args=[url]).start()
    print(f"\n  Deepfake HITL prototype running at {url}\n  Press CTRL+C to stop.\n")
    create_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
