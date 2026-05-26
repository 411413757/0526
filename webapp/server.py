import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
WEB_DIR = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)


def run_server():
    with socketserver.TCPServer(('0.0.0.0', PORT), Handler) as httpd:
        url = f'http://127.0.0.1:{PORT}/'
        print(f'Serving at {url}')
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()


if __name__ == '__main__':
    run_server()
