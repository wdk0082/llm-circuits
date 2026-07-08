# gcp/ — Cloud TPU lifecycle

Scripts to run `llm-circuits` on a Google Cloud TPU **without keeping one
running 24/7**. The TPU is disposable compute; durable state lives in a GCS
bucket in your own project. All scripts run on your **laptop** and load
config from `.env` via `lib.sh`.

```
laptop  ──gcloud──►  TPU VM (compute, ephemeral)  ──gs://──►  bucket (durable)
  ▲                                                              │
  └──────────────────── pull artifacts ─────────────────────────┘
```

## Projects & storage

| | project | role |
|---|---|---|
| compute | `dis-2026-tpu-zw499` (course-managed) | rents the v6e chip; **no storage here** |
| storage | `myloop-2026` (yours, your billing) | bucket `gs://dis-2026-zw499-tpu-store` (`us-east5`) |

Same region for both → no egress cost. The TPU reaches the cross-project
bucket via the auth wired by `setup_storage.sh`.

## One-time setup

```bash
gcp/setup_storage.sh     # grant the TPU's service account access to your bucket
```

(The bucket already exists; `setup_storage.sh` only does the IAM grant, and
asks first. It also prints an SA-key fallback if the keyless grant doesn't work.)

Also set in `.env`: `GIT_REMOTE` (the repo URL the VM clones — e.g.
`https://github.com/wdk0082/llm-circuits.git`) and your `CRSID`.

## Per-experiment loop

```bash
gcp/create.sh                          # provision (Spot + queued resource); waits for ACTIVE, bootstraps
git commit -am wip && git push         # the VM runs committed code (GIT_REF, default main)
gcp/launch.sh examples/addition_circuit.py   # run on the TPU; LLM_CIRCUITS_DEVICE=tpu + GCS dirs injected
gcp/pull.sh                            # sync gs://…/artifacts -> ./artifacts for inspection
gcp/teardown.sh                        # delete the TPU; data stays in the bucket
```

Helpers: `gcp/status.sh` (state of QR / TPU / bucket), `gcp/ssh.sh [cmd]`
(shell or one-off command on the VM).

## How a run is wired

- **`./bin/run`** is the entrypoint on both laptop and VM: it sources `.env`,
  prepends `.venv/bin` to `PATH`, then execs. `launch.sh` runs
  `./bin/run python -u <script.py>` on the VM.
- **Device:** `launch.sh` injects `LLM_CIRCUITS_DEVICE=tpu` (this repo's device
  env var, read by `llm_circuits.settings.default_device`). Wiring `tpu` to an
  actual `torch_xla` XLA device in `settings.py` / the model loaders is a
  deferred follow-up — the lifecycle plumbing already delivers the marker.
- **torch_xla** is installed on the VM by `bootstrap.sh`
  (`torch_xla[tpu]==2.10.0`, matched to the `torch` pin in `uv.lock`), kept out
  of `pyproject.toml` so the lockfile stays cross-platform. Override with
  `TORCH_XLA_VERSION` in `.env` if you bump `torch`.
- **Artifacts** are written to `$LLM_CIRCUITS_ARTIFACTS_DIR` (staged to
  `~/scratch/artifacts` on the VM). In bucket mode `launch.sh` rsyncs them to
  `gs://<bucket>/artifacts` after the run; in local-pull mode they are scp'd
  back to `./artifacts`.

## Conventions

- **Spot by default** (`TPU_SPOT=1`): cheap and preemptible. Make training
  resumable — checkpoint to `$CKPT_DIR` (GCS) every ~15–30 min and restore
  the latest on start. Set `TPU_SPOT=0` for on-demand.
- **Queued resources** are the provisioning unit (`QR_NAME`); deleting the
  queued resource deletes the node.
- **Local-pull mode**: leave `GCS_BUCKET` empty in `.env` to skip the bucket
  entirely — `launch.sh`/`teardown.sh` then `scp` artifacts to your laptop
  before deleting. Good for quick on-demand runs; weaker under preemption.
- **Never commit credentials.** SA keys live under `gcp/keys/` (gitignored).
- `WORKER=0` by default (single-host `v6e-1`); set `WORKER=all` for
  multi-host slices.
- `FORCE=1` skips confirmation prompts (for scripting teardown).

## Notes / TODO

- `launch.sh` passes experiment args positionally; quote complex args.
- Code reaches the VM via `git clone`/`pull` (push before launching). For
  fast local iteration without pushing, `scp` the changed files with
  `gcp/ssh.sh` or add an rsync helper.
- The on-VM checkpoint-to-GCS write + SIGTERM-on-preemption handler is
  framework-specific — add it to `src/llm_circuits/` alongside the torch_xla
  device wiring.
