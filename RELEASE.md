# Release Guide

Releases are fully automated by the
[`Release`](.github/workflows/release.yml) GitHub Actions workflow. Every
push to `main` produces a new patch release — no tagging, no version-file
edits, no manual upload. A typical release looks like:

```sh
git commit -m "fix: whatever"
git push origin main
# ~5-10 min later, a new vX.Y.(Z+1) release exists with the installer attached
```

## How it works

On every push to `main` (or a manual `workflow_dispatch` run):

1. **Version computation.** Reads all `v*` tags, picks the highest one by
   version sort, and bumps the patch component (`v0.2.3` → `v0.2.4`). If
   there are no tags yet, starts at `v0.0.1`.
2. **Version injection.** Patches `Cargo.toml` + `Cargo.lock` in the CI
   workspace so `CARGO_PKG_VERSION` baked into the binary matches the tag.
   `main` is never modified — the patch is ephemeral.
3. **Dependency fetch.** Downloads `buffalo_l.zip` from the InsightFace
   model zoo and SHA-256-verifies the archive plus the extracted
   `det.onnx` / `rec.onnx`. Also pulls a static `ffmpeg.exe` from
   [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds).
4. **Build.** `cargo build --release --locked`.
5. **Package.** Runs `ISCC /DMyAppVersion=X.Y.Z setup.iss` to produce
   `installer_output/Facial-Recognition-Sorter-X.Y.Z-Setup.exe`.
6. **Publish.** Uses `softprops/action-gh-release` to atomically create
   the `vX.Y.Z` tag at the triggering commit **and** a GitHub release with
   the installer attached and auto-generated notes from the commit range.

Multiple pushes in flight are serialized by a `release` concurrency group
so the second push computes its next version *after* the first push has
tagged, avoiding collisions.

## Minor / major bumps

Since the workflow always patch-bumps, a minor or major bump requires a
one-off hand tag:

```sh
git tag v0.3.0
git push origin v0.3.0
```

The next auto-bumping push after that will pick it up as the new baseline
(`v0.3.0` → `v0.3.1`).

A more polished alternative would be to add a `workflow_dispatch` input
for `bump: patch|minor|major`; wire it up when you want it.

## Skipping a release

If you want to push to `main` without cutting a release (docs typo, CI
tweak, etc.), add `[skip ci]` to the commit message. GitHub Actions
respects that marker and skips the workflow entirely.

## Re-running after a failure

Open the Actions tab, find the failed run, and click **Re-run jobs**. The
version recomputes, so a transient flake (network, runner outage)
shouldn't waste a version number in most cases.

## Local installer builds (reference)

The workflow is the recommended path, but you can still build the
installer locally for testing:

### 1. Build the app

```powershell
cargo build --release
```

### 2. Stage FFmpeg

The installer bundles `ffmpeg.exe`. Place it at:

`third_party/ffmpeg/ffmpeg.exe`

### 3. Stage the SCRFD + ArcFace models

The installer bundles `models/det.onnx` + `models/rec.onnx`. These come
from InsightFace's `buffalo_l` pack:

```
https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
```

Extract `det_10g.onnx` → `models/det.onnx` and `w600k_r50.onnx` →
`models/rec.onnx`.

### 4. Build installer

```powershell
iscc /DMyAppVersion=0.2.4 setup.iss
```

If you skip `/DMyAppVersion=`, the script falls back to `0.0.0-dev`.

### 5. Signed installer (optional)

`setup.iss` supports signing when `FRS_SIGN_CMD` is set:

```powershell
$env:FRS_SIGN_CMD='signtool sign /n "Your Cert Subject" /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com'
iscc /DMyAppVersion=0.2.4 setup.iss
```

CI currently does **not** sign installers. To enable signing in CI, add
the signing command as a repository secret and extend
`.github/workflows/release.yml` to export `FRS_SIGN_CMD` before the
"Build installer" step.

### 6. Verify icon quality

- Keep `res/app_icon.ico` as a true multi-resolution ICO.
- Include at minimum: `16, 20, 24, 32, 40, 48, 64, 96, 128, 256`.

Quick check (PowerShell):

```powershell
$bytes=[System.IO.File]::ReadAllBytes('res/app_icon.ico')
$count=[BitConverter]::ToUInt16($bytes,4)
for($i=0;$i -lt $count;$i++){
  $o=6+$i*16
  $w=$bytes[$o]; if($w -eq 0){$w=256}
  $h=$bytes[$o+1]; if($h -eq 0){$h=256}
  "$i : ${w}x${h}"
}
```
