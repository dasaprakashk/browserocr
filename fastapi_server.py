from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


PORT = 8080
ROOT_DIR = Path(__file__).resolve().parent


class HeaderStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        request_path = scope.get("path", "")

        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Access-Control-Allow-Origin"] = "*"

        if request_path.endswith(".onnx"):
            response.headers["Cache-Control"] = "public, max-age=86400"
        else:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response


app = FastAPI(title="Costco OCR Server")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index(_: Request) -> FileResponse:
    return FileResponse(ROOT_DIR / "index.html")


app.mount("/", HeaderStaticFiles(directory=ROOT_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    print(f"FastAPI OCR server running at http://localhost:{PORT}")
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=PORT, reload=False)