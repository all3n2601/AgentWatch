from fastapi import FastAPI

app = FastAPI(title="Model Passport Worker", version="0.1.0")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "worker"}

