# Facial Recognition Sorter

A fast desktop app to find one person across a big photo/video library.

---

## Quick Install 🚀

1. **Download the Installer:** Go to the [Releases](https://github.com/davidgiang1/facial-recognition-sorter/releases) page and download the latest `Facial-Recognition-Sorter-<version>-Setup.exe`.
2. **Install:** Run the setup file.
3. **Run:** Open "Facial Recognition Sorter" from your Start Menu.

The installer already includes AI models + `ffmpeg`.

---

## Quick Start 📸

1. Select **Input Directory** (the big folder to search).
2. Select **People Library** (`people/`).
3. Pick **Target Person** from the dropdown.
4. Click **Start Processing**.
5. Copy good matches into that person’s folder.

Expected layout:

```
people/
  Person_One/
    ref1.jpg
    ref2.jpg
  Person_Two/
    ref1.jpg
```

Tip: first run is slower; later runs are much faster. ⚡

---

## FAQ ❓

**Does this require an internet connection?**
No. Everything runs locally on your computer.

**What kind of graphics card do I need?**
Most modern Windows GPUs are supported (NVIDIA, AMD, Intel via DirectML).

---

## For Developers 🛠️

- Build + architecture: [DEVELOPMENT.md](DEVELOPMENT.md)
- Packaging + release flow: [RELEASE.md](RELEASE.md)

---

## License

[MIT](LICENSE)
