"""
ws_test_client.py – Test the WebSocket endpoint locally.

Usage:
    python ws_test_client.py path/to/audio.mp3
    python ws_test_client.py path/to/audio.mp3 --url ws://localhost:8000/ws/process
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path


async def run(audio_path: str, url: str) -> None:
    try:
        import websockets
    except ImportError:
        print("Install websockets:  pip install websockets")
        sys.exit(1)

    path = Path(audio_path)
    if not path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    audio_b64 = base64.b64encode(path.read_bytes()).decode()

    print(f"Connecting to {url}")
    print(f"Uploading: {path.name}  ({path.stat().st_size / 1024:.1f} KB)\n")

    async with websockets.connect(url) as ws:
        # Send the job
        await ws.send(json.dumps({
            "filename":  path.name,
            "audio_b64": audio_b64,
        }))

        # Stream events
        async for raw in ws:
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "progress":
                step  = msg["step"]
                total = msg["total"]
                bar   = "█" * step + "░" * (total - step)
                print(f"[{bar}] {step}/{total}  {msg['message']}")

            elif event == "error":
                print(f"\n❌  ERROR at step {msg['step']}: {msg['message']}")
                break

            elif event == "result":
                print("\n✅  Pipeline complete!")
                print(f"   Processed at : {msg.get('processed_at')}")
                print(f"   Context      :\n     " +
                      "\n     ".join(msg.get("context_info", "").splitlines()))
                print(f"\n   Output files :")
                for k, v in msg.get("output_files", {}).items():
                    print(f"     {k:10s}  {v}")
                print(f"\n   Transcription preview (first 3 lines):")
                for line in msg.get("original_transcription", "").splitlines()[:3]:
                    print(f"     {line}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket test client for transCallMapping")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--url", default="ws://localhost:8000/ws/process")
    args = parser.parse_args()
    asyncio.run(run(args.audio, args.url))