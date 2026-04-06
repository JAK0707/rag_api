# ============================================================
# FILE: scripts/profile_query.py
# PURPOSE: cProfile a full upload + query cycle; print top 20 hotspots
# ============================================================

"""
Usage:
    OPENAI_API_KEY=sk-... python scripts/profile_query.py path/to/sample.pdf

Requires a running Redis instance and valid OPENAI_API_KEY.
Output is printed to stdout and saved to profile_results.txt.
"""

import asyncio
import cProfile
import io
import os
import pstats
import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"


async def run_upload_and_query(pdf_path: str) -> None:
    """Perform a single upload + query cycle against the local server."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        # --- Upload ---
        with open(pdf_path, "rb") as fh:
            upload_resp = await client.post(
                "/api/v1/upload",
                files={"file": (os.path.basename(pdf_path), fh, "application/pdf")},
            )
        upload_resp.raise_for_status()
        doc_id = upload_resp.json()["doc_id"]
        print(f"Uploaded — doc_id: {doc_id}, chunks: {upload_resp.json()['chunks_count']}")

        # --- Query ---
        query_resp = await client.post(
            "/api/v1/query",
            json={"doc_id": doc_id, "question": "Summarise the main topic of this document."},
        )
        query_resp.raise_for_status()
        result = query_resp.json()
        print(f"Answer: {result['answer'][:200]}")
        print(f"Latency: {result['latency_ms']} ms | Cached: {result['cached']}")


def profile_main(pdf_path: str) -> None:
    """Run the async cycle under cProfile and report the top 20 functions."""
    profiler = cProfile.Profile()
    profiler.enable()

    asyncio.run(run_upload_and_query(pdf_path))

    profiler.disable()

    string_buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=string_buffer)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    output = string_buffer.getvalue()
    print("\n--- cProfile top 20 (cumulative time) ---")
    print(output)

    output_file = "profile_results.txt"
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(output)
    print(f"Profile results saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_query.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    profile_main(pdf_path)
