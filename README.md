# Facial Recognition Sorter

A desktop application that uses artificial intelligence to find photos of a specific person within a large collection of images. 

Given a folder of **target reference images** (photos of the person you are looking for) and a large **input image directory** (the collection to search), the application will quickly scan all the images, find matches, and allow you to easily browse and export them.

---

## Installation (No Code Required)

To use the application without any technical setup, follow these steps:

1. **Download the Installer:** Go to the [Releases](https://github.com/davidgiang1/facial-recognition-sorter/releases) page and download the latest `Facial-Recognition-Sorter-Setup.exe`.
2. **Install:** Double-click the downloaded setup file to install the application to your computer.
3. **Run:** Open "Facial Recognition Sorter" from your Start Menu.

*(Note: The installer automatically includes the necessary AI models, so you don't need to download them separately!)*

---

## How to Use

1. **Launch the application.**
2. **Select Target Directory:** Click "Browse" to choose a folder containing a few clear reference photos of the person you want to find.
3. **Select Input Directory:** Click "Browse" to choose the large folder of images you want to search through.
4. **Adjust Threshold:** Use the distance threshold sliders if needed (a lower number means a stricter match, but might miss some photos; a higher number finds more photos, but might include incorrect people).
5. **Search:** Click the **Search** button. 
6. **Review Results:** The app will process the images and show you the best matches. You can click on thumbnails to view them, select the ones you want, and use the "Copy selected to output" button to save them to a new folder!

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
