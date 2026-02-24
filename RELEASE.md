# Release Guide

## 1. Build the app

```powershell
cargo build --release
```

## 2. Verify icon quality before packaging

- Keep `res/app_icon.ico` as a true multi-resolution ICO.
- Include at minimum: `16, 20, 24, 32, 40, 48, 64, 96, 128, 256`.
- Ensure the small sizes are hand-tuned exports (not only auto-downscaled).

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

## 3. Build installer (unsigned)

```powershell
iscc setup.iss
```

## 4. Build installer (signed)

`setup.iss` supports signing when `FRS_SIGN_CMD` is set.

Example with `signtool` and timestamping:

```powershell
$env:FRS_SIGN_CMD='signtool sign /n "Your Cert Subject" /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com'
iscc setup.iss
```

Notes:
- This signs installer binaries (`$f`) and the uninstaller.
- If `FRS_SIGN_CMD` is not set, installer builds unsigned.

## 5. Publish GitHub release

1. Create/update a tag (example: `v0.1.1`).
2. Upload `installer_output/Facial-Recognition-Sorter-Setup.exe`.
3. Add release notes with changes and any known limitations.
