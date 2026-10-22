# InsightFace Server Web UI

This directory is a build-free, same-origin browser client for the public
`/v1` REST API. The server mounts these files at `/assets`, serves `index.html`
at `/`, and serves `openapi.html` at `/docs`. The Help area renders the bundled
User Guide or complete API Guide in all nine supported languages, plus the
English-only Maintainer Guide. A document switcher displays one source at a
time with its own searchable table of contents. These are the same Markdown
files linked from GitHub; `/docs` remains the live, schema-derived API
reference. Repository READMEs are not served by the application.

Explicit document links preserve their target language, including unsuffixed
English Markdown files; same-page fragments keep the displayed language. The
language selector follows a localized document link. The Maintainer Guide
remains English without changing the console language.

Liveness is disabled by default. The dashboard always reports its active state;
System separates that state from the verified model file and saved restart
setting. When the deployment permits writes, the Web action downloads or reuses
`/models/addons/liveness.onnx`, saves the selected startup configuration, and
requests a manual Server restart. Polling resumes when the page is reopened.
Unavailable deployments and failed jobs show localized advice using stable API
codes, with model/configuration paths kept separate from translated messages.
The page does not switch base model packages.

The UI uses only browser-native HTML, CSS, JavaScript modules, images, and canvas.
It has no package-manager, CDN, analytics, or third-party font
dependency. The API key is held in a private JavaScript field for the lifetime
of the current tab; it is never written to browser storage, cookies, or URLs.

Camera Monitoring administers persistent server-side RTSP Monitors through the
public `/v1` API. Closing the page never stops a task. The UI polls signed-cursor
in-memory events and live state; optional raw MJPEG preview is off by default
and encoded only while viewed. The browser draws green enrolled and amber
unknown-face overlays from `/state`. It never requests local camera permission
or stores frames, events, RTSP URLs, credentials, or API keys.

Run the browser-independent checks from the repository root:

```bash
node --test server/tests/frontend/*.test.mjs
```
