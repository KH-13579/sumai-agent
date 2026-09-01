"""スマイエージェント — FastAPI アプリケーション本体"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# .env 読み込み
load_dotenv()

from app.api.chat import router as chat_router
from app.api.artifact import router as artifact_router

app = FastAPI(
    title="スマイエージェント",
    description="マルチAIエージェント型 住宅意思決定支援サービス",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API ルーター
app.include_router(chat_router, prefix="/api")
app.include_router(artifact_router, prefix="/api")

# フロントエンド静的ファイル配信
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", response_class=FileResponse)
    async def serve_index():
        return FileResponse(str(frontend_dir / "index.html"))
