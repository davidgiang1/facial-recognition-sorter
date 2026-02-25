# Facial Recognition Sorter

A desktop application that uses artificial intelligence to find photos of a specific person within a large collection of images. 

Given a folder of **target reference images** (photos of the person you are looking for) and a large **input image directory** (the collection to search), the application will quickly scan all the images, find matches, and allow you to easily browse and export them.

---

## Installation (No Code Required)

To use the application without any technical setup, follow these steps:

1. **Download the Installer:** Go to the [Releases](https://github.com/davidgiang1/facial-recognition-sorter/releases) page and download the latest `Facial-Recognition-Sorter-Setup.exe`.
2. **Install:** Double-click the downloaded setup file to install the application to your computer.
3. **Run:** Open "Facial Recognition Sorter" from your Start Menu.

*(Note: The installer automatically includes the necessary AI models and `ffmpeg`, so you don't need to install them separately.)*

---

## How to Use

1. **Launch the application.**
2. **Select Input Directory:** Choose the large folder of images/videos you want to search.
3. **Select People Library:** Choose a parent folder (for example `people/`) that contains one subfolder per person.
4. **Select Target Person:** Pick the person subfolder from the dropdown (for example `people/Alice/`).
5. **Adjust Thresholds:** Tune distance/rejection sliders if needed.
6. **Start Processing:** Click the **Start Processing** button.
7. **Review Results:** The app shows ranked matches. Select any results and use "Copy Selected to Person Folder" to add them to the current person.

Expected folder structure:

```
people/
  Alice/
    ref1.jpg
    ref2.jpg
  Bob/
    ref1.jpg
```

*Note: The first time you search a large folder, it may take some time as it analyzes all the faces. Subsequent searches in the same folder will be much faster.*

---

## FAQ

**Does this require an internet connection?**
No. All processing is done locally on your computer. No images or data are ever uploaded to the internet. Your privacy is guaranteed.

**What kind of graphics card do I need?**
The application uses DirectML, which works on most modern Windows GPUs (NVIDIA, AMD, and Intel).

---

## Development & Technical Details

Are you a developer interested in how this application is built, its architecture, or how to compile it from source? Please refer to the [DEVELOPMENT.md](DEVELOPMENT.md) guide.

For release packaging, installer signing, and icon quality checks, use [RELEASE.md](RELEASE.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
