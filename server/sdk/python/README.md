# InsightFace Server Python client

The lightweight client for the self-hosted InsightFace Server REST API. It uses
`httpx`, contains no inference runtime, and accepts image paths, bytes, or binary
file-like objects.

```bash
python -m pip install ./server/sdk/python
```

```python
from insightface_server import Client

with Client("http://localhost:8080", api_key="replace-me") as client:
    faces = client.detect("photo.jpg")
    print(faces.faces)
```

The system detection profile is startup-only. A Collection copies it at
creation and may override input sizes, detector/NMS thresholds, and the
`largest` or `center_largest` single-face strategy. Pass `collection=` to
stateless Detect, Compare, or Embeddings calls to use that Collection profile.

Trusted upstream extractors may pass `external_embeddings` together with the
required images and the Collection's `embedding_contract_id`. This selects
`external_trusted`: image detection and quality review still run, while the
server neither re-extracts nor falls back to another feature.

Persistent RTSP monitoring is also available through `create_monitor`,
`update_monitor`, `monitor_state`, and cursor-based `monitor_events`. Monitor
preview is off by default; recognition and in-memory events do not require it.
The client waits up to 65 seconds by default, slightly longer than the server's
60-second request deadline. Pass `timeout=` to `Client` when an application
needs a different fail-fast policy.

See `server/docs/user-guide.md` for complete SDK and operating workflows, and
`server/docs/api.md` for the full HTTP contract.
