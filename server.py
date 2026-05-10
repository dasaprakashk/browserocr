"""
server.py
Local HTTP server with CORS and COOP/COEP headers for onnxruntime-web.

Sets Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
which are required by onnxruntime-web for WASM loading in modern browsers.

Usage:
    python server.py
Then open http://localhost:8082 in your browser.
"""

import http.server
import os
import socketserver

PORT = int(os.environ.get('PORT', '8082'))

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required for SharedArrayBuffer / WASM threading
        self.send_header('Cross-Origin-Opener-Policy',   'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin',  '*')

        # Models (large .onnx files) are cached in IndexedDB by the app,
        # so we only no-store for JS/HTML/CSS to ensure code changes reload.
        path = self.path.split('?')[0]
        if path.endswith('.onnx'):
            # Allow browser to cache the raw response too (speeds up first load on revisit)
            self.send_header('Cache-Control', 'public, max-age=86400')
        else:
            # JS/HTML/CSS: never cache so code changes are always picked up
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma',        'no-cache')
            self.send_header('Expires',       '0')
        super().end_headers()

    def log_message(self, format, *args):
        # Show all requests so we can see what the browser loads
        super().log_message(format, *args)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(('', PORT), CORSRequestHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f'✅ Costco OCR server running at http://localhost:{PORT}')
        print('   Press Ctrl+C to stop.\n')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nServer stopped.')
