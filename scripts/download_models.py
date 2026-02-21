import os
import urllib.request
import shutil
from pathlib import Path

# Direct download for DINOv2
MODELS = {
    "dinov2_vits14.onnx": "https://huggingface.co/sefaburak/dinov2-small-onnx/resolve/main/dinov2_vits14.onnx"
}

def main():
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ensuring models are present in {models_dir}...")

    # 1. Download DINOv2 (and any other direct links)
    for filename, url in MODELS.items():
        filepath = models_dir / filename
        if filepath.exists():
            print(f"[-] {filename} already exists, skipping.")
            continue

        print(f"[*] Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            print(f"[+] Successfully downloaded {filename}")
        except Exception as e:
            print(f"[!] Error downloading {filename}: {e}")

    # 2. Safely Generate YOLO26m.onnx using the official library (NMS-free, end-to-end)
    yolo_path = models_dir / "yolo26m.onnx"
    if not yolo_path.exists():
        print(f"[*] Generating yolo26m.onnx via official ultralytics package...")
        try:
            from ultralytics import YOLO

            # YOLO26-M: NMS-free architecture, faster inference, simplified ONNX export
            model = YOLO("yolo26m.pt")

            # YOLO26 is natively NMS-free, no special flags needed
            exported_path = model.export(format="onnx", dynamic=False, opset=13)

            # Move the exported file to our models directory
            shutil.move(exported_path, yolo_path)

            # Cleanup the leftover .pt file since we only need the ONNX version for Rust
            if os.path.exists("yolo26m.pt"):
                os.remove("yolo26m.pt")

            print("[+] Successfully generated yolo26m.onnx!")

        except ImportError:
            print("\n[!] Error: The 'ultralytics' python package is missing.")
            print("[!] Please run: pip install ultralytics")
            print("[!] Then run this script again.")
        except Exception as e:
            print(f"\n[!] Error exporting YOLO: {e}")
    else:
        print("[-] yolo26m.onnx already exists, skipping.")

if __name__ == "__main__":
    main()