# DTU Aqua Multitool

A modern, user-friendly web application for advanced fish detection and measurement using deep learning (YOLO segmentation). Built with Streamlit for fast, interactive analysis in research, field, or educational settings.

---

## Features
- **Batch Processing:** Upload and analyze multiple images or videos at once.
- **Field Mode:** Quick, single-image analysis for field work.
- **Live Camera:** Real-time fish detection and measurement using your device's camera.
- **Adjustable Confidence:** Fine-tune detection sensitivity.
- **Downloadable Results:** Export detection results as CSV.
- **Modern UI:** Clean, professional design with helpful mode descriptions and visual cues.

---

## Setup

1. **Install Git LFS:**
   
   Windows:
   ```cmd
   winget install Git.Git.LFS
   # or with Chocolatey
   choco install git-lfs
   ```
   
   Mac:
   ```bash
   brew install git-lfs
   ```
   
   Linux:
   ```bash
   sudo apt install git-lfs  # Ubuntu/Debian
   sudo yum install git-lfs  # CentOS/RHEL
   ```

2. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd DTU-Multitool
   git lfs pull  # Pull the YOLO model file
   ```

3. **Create and activate a virtual environment:**
   
   Windows:
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
   Mac/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

The YOLO model (`models/yolo11m-seg.pt`) will be automatically downloaded during the Git LFS pull.

---

## Usage

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   - The app will automatically open in your default browser
   - Or go to the local URL provided by Streamlit (usually `http://localhost:8501`)

3. **Choose a mode:**
   - **Batch Processing:** For multiple files.
   - **Field Mode:** For single, quick analysis.
   - **Live Camera:** For real-time detection.

4. **Upload images/videos or use your camera.**
5. **View and download results.**

---

## Troubleshooting
- **Git LFS Issues:**
  - Run `git lfs install` to initialize LFS
  - Use `git lfs pull` to manually download the model file
  - Check `.gitattributes` if model isn't downloading
- **Virtual Environment Not Activating:**
  - Windows: Try running as administrator or use `py -m venv venv`
  - Mac/Linux: Ensure you have permissions (`chmod +x venv/bin/activate`)
- **Missing Packages:**
  - Make sure your virtual environment is activated and all dependencies are installed.
- **Model Not Found:**
  - Verify Git LFS pulled the model file correctly
  - Check the model file exists in the models directory
- **Images Not Displaying:**
  - Check your internet connection for online images, or use local images if needed.
- **Linter Errors in Editor:**
  - Select the correct Python interpreter (your venv) in your IDE.

---

## Credits
- Built with [Streamlit](https://streamlit.io/) and [Ultralytics YOLO](https://ultralytics.com/).
- Fish icons from [OpenMoji](https://openmoji.org/).

---

## License
MIT License
