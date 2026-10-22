# Changelog

## 0.3.1

- Simplify CPU and CUDA Docker deployment: Server and model installer run as
  root (`0:0`) with Docker's default capabilities, without `cap_drop: [ALL]`.
  The container root filesystem remains read-only and `no-new-privileges` stays
  enabled; privileged mode is not required.
- Use one writable `/models` bind mount in each service and let Compose create
  a missing model root. Remove the separate addon mount and host UID/GID,
  shared-group, and manual addon-directory preparation steps. Explicit addon
  downloads create their own subdirectory; liveness stays disabled by default,
  and Server startup never downloads models.
- Add `models install <package> --enable-liveness` to download and verify the
  required models, then enable liveness in the shared startup configuration,
  including cache hits and installation before the first Server startup. Preserve
  other addon entries, settings, and comments; fail explicitly if configuration
  cannot be prepared or saved. Downloads must succeed before configuration changes.
- Mount the whole existing configuration directory writable in both Compose
  services for atomic Web/CLI configuration saves. Plain model installation and
  `models addons install liveness` remain configuration-preserving; an already
  running Server must be restarted to apply saved activation.
- Update all localized deployment, upgrade, and API installation instructions.
  Upgrades must apply the new Compose settings and custom overrides as well as
  the image tags, preserving existing models, configuration, and data volumes.
- Keep the Server and Python SDK versions aligned at `0.3.1`. No public REST
  operation, response contract, or embedding contract changes in this release.

## 0.3.0

- Add `raccoon_s` and `raccoon_l` detector/recognizer packages with support for
  their model manifests; the PrivateFrame verifier is not loaded by Server.
- Add optional liveness, explicit Web model installation and next-start
  configuration, and BMP image input. Liveness defaults to disabled.
- Remove the separate `model_version` response field; model identity uses
  `model_id`, and Collection compatibility uses `embedding_contract_id`.
