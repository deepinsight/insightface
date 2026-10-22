# InsightFace Server Python client

The lightweight client for the self-hosted InsightFace Server REST API. It uses
`httpx`, contains no inference runtime, and accepts image paths, bytes, or binary
file-like objects. JPEG, PNG, WebP, and BMP inputs are supported by every
image operation, including enrollment. Saved face crops remain JPEG.

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

Model packages are identified by `model_id`, without a separate `model_version`
field. Read `embedding_contract_id` from the target Collection when submitting
external embeddings; existing Collection identifiers survive the upgrade.

## Liveness results and errors

The distributed Server configuration disables liveness by default; enabling it is
configured on the Server, not per SDK request. The Web UI can download the addon
and save its activation in `server.toml`; an operator must restart Server before
it affects SDK calls. See the [Server user guide](../../docs/user-guide.md#optional-liveness-addon). Access
`client.detect(image).faces[0].get("liveness")`; when evaluated, the mapping has
the three core fields `status`, `is_live` and `live_score`. Absence means not
evaluated. Only insufficient image area around the aligned face produces
`status == "input_rejected"`, with `is_live` and `live_score` set to `None` and
an additional human-readable `reason`. Live and fake results omit `reason`.
The API always returns this explanation in English:

> Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image.

Use `status` and `is_live` for program logic; `reason` is not an enumeration code.
Older saved results may lack `reason`, so keep a generic display fallback.

In `normal`, compare, embeddings and search raise `ValidationError` (HTTP 422)
with `code == "liveness_fake"` or `"liveness_input_rejected"`. Read
`error.details["liveness"]`, and for comparison `error.details["side"]`.
Enrollment skips liveness by default (`[inference].liveness_on_registration=false`),
for both new Persons and added FaceSamples; new samples then omit `liveness`.
With that server setting enabled, enrollment follows `normal`/`observe`, and its
liveness rejection entries expose the same mapping. An all-rejected new Person
uses `registration_failed` with `details["rejected_images"]`. Runtime inference
failures raise `ServiceUnavailableError` with `code == "liveness_unavailable"`
and stop the operation in both `normal` and `observe`; they are not input rejections.
In `observe`, recognition continues after live, fake or input-rejected results,
and successful face results retain liveness.

```python
from insightface_server import ValidationError

try:
    result = client.compare(source, target)
except ValidationError as error:
    if error.code in {"liveness_fake", "liveness_input_rejected"}:
        liveness = error.details["liveness"]
        print(error.details["side"], liveness)
        if liveness["status"] == "input_rejected":
            print(liveness.get("reason") or "Input rejected by liveness detection.")
    else:
        raise
```
