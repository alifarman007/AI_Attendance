# AI Attendance System

A complete Python application for automated attendance tracking using face recognition technology. The system uses state-of-the-art face detection (SCRFD) and face recognition (ArcFace) models with FAISS for fast face identification.

## Features

- **Real-time Face Detection**: Uses SCRFD (Sample and Computation Redistribution for Efficient Face Detection) for accurate face detection
- **Face Recognition**: Employs ArcFace ResNet-50 for robust face recognition with high accuracy
- **Fast Face Identification**: FAISS (Facebook AI Similarity Search) for efficient face matching
- **Attendance Logging**: Automatic Excel-based attendance logging with duplicate prevention
- **Person Management**: Easy system for adding new people to the database
- **Real-time Monitoring**: Live webcam feed with face recognition overlay
- **Statistics & Reporting**: Built-in attendance statistics and export functionality

## Requirements

### Hardware
- Webcam or USB camera
- CPU with at least 4GB RAM (GPU not required)

### Software
- Python 3.8 or higher
- Required Python packages (see requirements.txt)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd AI_Attendance
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Files

You need to manually download the following model files and place them in the `weights/` directory:

| Model | File Name | Size | Download Link | Purpose |
|-------|-----------|------|---------------|---------|
| SCRFD 10G | `det_10g.onnx` | ~16.1 MB | [Download](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx) | Face Detection |
| ArcFace ResNet-50 | `w600k_r50.onnx` | ~166 MB | [Download](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | Face Recognition |

**Important**: Create the `weights/` directory and place both model files there:

```bash
mkdir weights
# Download the files and place them in weights/
# weights/det_10g.onnx
# weights/w600k_r50.onnx
```

### 5. Directory Structure

After setup, your directory structure should look like this:

```
AI_Attendance/
├── weights/
│   ├── det_10g.onnx          # Face detection model
│   └── w600k_r50.onnx        # Face recognition model
├── faces/                    # Person photos go here
│   ├── john_doe.jpg          # Example person photo
│   └── jane_smith.jpg        # Another person photo
├── database/                 # FAISS database files (auto-generated)
├── models/                   # Model wrapper classes
├── utils/                    # Utility functions
├── main.py                   # Main application
├── face_recognition.py       # Face recognition logic
├── attendance_logger.py      # Attendance logging
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

### Adding People to the System

Before running the attendance system, you need to add people to the face database:

#### Method 1: Add Photos to faces/ Directory

1. Take a clear photo of each person (good lighting, facing camera)
2. Save the photo as `firstname_lastname.jpg` in the `faces/` directory
3. The filename becomes the person's name in the system

Example:
```bash
faces/
├── john_doe.jpg
├── jane_smith.jpg
├── alice_johnson.jpg
└── bob_wilson.jpg
```

#### Method 2: Command Line Addition

```bash
python main.py --add-person path/to/photo.jpg "Person Name"
```

#### Method 3: Interactive Addition

1. Run the main system: `python main.py`
2. Press 'a' during operation
3. Follow the prompts to add person photos

### Running the Attendance System

You can run the attendance system in two ways:

#### Option 1: Gradio Web Interface (Recommended)

```bash
python Gradio_interface.py
```

This launches a user-friendly web interface with:
- **👤 Add Person to Database Tab**: 
  - Upload photos directly through browser
  - Add people to the face recognition database
  - Real-time database status updates
  - Automatic face detection validation
- **📹 Live Attendance View Tab**: 
  - Start/Stop webcam controls
  - Real-time face recognition overlay
  - Auto-updating attendance table
  - Today's attendance summary
  - Download attendance Excel file

**Features:**
- ✅ **Real-time Updates**: Attendance table refreshes automatically every 2 seconds
- ✅ **Face Validation**: System validates uploaded images contain detectable faces
- ✅ **Auto-restart Recognition**: Adding new people automatically updates the recognition pipeline
- ✅ **Live Statistics**: Real-time display of today's attendance summary
- ✅ **Excel Export**: One-click download of attendance data
- ✅ **Responsive Design**: Works on desktop and mobile browsers

The web interface will be available at `http://localhost:7860` and will automatically open in your browser.

#### Option 2: Command Line Interface

```bash
python main.py
```

This starts the traditional CLI version with real-time attendance monitoring.

#### Advanced Usage

```bash
# Use a different camera
python main.py --camera 1

# Show attendance statistics only
python main.py --stats

# Export attendance data
python main.py --export
```

### System Controls

#### Gradio Web Interface Controls

**👤 Add Person to Database Tab:**
- Upload image files directly through the browser (supports .jpg, .png, .jpeg)
- Enter person's full name and click "➕ Add to Database"
- Real-time database statistics showing total faces and unique people
- Automatic face detection validation (warns if no face detected)
- Auto-saves images to faces/ directory for future use

**📹 Live Attendance View Tab:**
- **▶️ Start Webcam** / **⏹️ Stop Webcam** buttons for camera control
- Real-time webcam feed with face detection/recognition overlay
- Auto-updating attendance table (refreshes every 2 seconds)
- Today's attendance summary with live statistics
- **📥 Download Attendance Excel** button for instant data export
- System info overlay showing detection statistics and current time

#### CLI Controls (main.py)

While the CLI system is running:

- **'q'**: Quit the application
- **'a'**: Add new person interactively
- **'s'**: Show current statistics
- **ESC**: Also quits the application

### Understanding the Output

The system shows:
- **Green boxes**: Known people with their names and similarity scores
- **Red boxes**: Unknown faces
- **System info**: Frame count, detected faces, recognized faces, current time
- **Console logs**: Attendance logging notifications and system status

## How It Works

### 1. Face Detection (SCRFD)
- Detects faces in each video frame
- Provides bounding boxes and facial landmarks
- Optimized for real-time performance

### 2. Face Recognition (ArcFace)
- Extracts 512-dimensional face embeddings
- Uses facial landmarks for proper alignment
- Highly accurate face representation

### 3. Face Identification (FAISS)
- Stores face embeddings in a searchable vector database
- Performs fast similarity search using cosine distance
- Enables real-time face matching

### 4. Attendance Logging
- Records name, date, time, and similarity score
- Prevents duplicate entries within the same day (configurable)
- Saves data to Excel file (`attendance.xlsx`)

## Configuration

Edit `config.py` to customize the system:

```python
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5    # Face detection confidence
SIMILARITY_THRESHOLD = 0.4    # Face recognition similarity
DUPLICATE_THRESHOLD_HOURS = 8 # Hours before re-logging same person

# Camera settings  
CAMERA_INDEX = 0              # Which camera to use

# File paths
ATTENDANCE_FILE = "attendance.xlsx"  # Output file name
```

## Attendance Data

The system creates an Excel file (`attendance.xlsx`) with columns:
- **Name**: Person's name
- **Date**: Date of attendance  
- **Time**: Time of attendance
- **Similarity_Score**: Face recognition confidence (0-1)

### Duplicate Prevention

The system prevents duplicate attendance logging for the same person within a configurable time window (default: 8 hours). This prevents multiple logs if someone passes by the camera multiple times.

## Troubleshooting

### Common Issues

1. **"Cannot open camera"**
   - Check if your camera is being used by another application
   - Try different camera indices (0, 1, 2, etc.)
   - Ensure camera permissions are granted
   - In Gradio interface: Close any other camera applications and restart

2. **"Failed to load model weights"**
   - Verify model files are in the `weights/` directory
   - Check file names match exactly: `det_10g.onnx` and `w600k_r50.onnx`
   - Ensure files downloaded completely

3. **"No face detected in image"** (especially when adding people via Gradio)
   - Use clear, well-lit photos
   - Ensure face is clearly visible and not too small
   - Face should be front-facing and unobstructed
   - Try different photos of the same person
   - The Gradio interface will show this error message if no face is found

4. **Poor recognition accuracy**
   - Adjust `SIMILARITY_THRESHOLD` in config.py
   - Use higher quality reference photos
   - Ensure good lighting during live operation

5. **Gradio interface issues**
   - **Browser not opening automatically**: Navigate manually to `http://localhost:7860`
   - **Interface not loading**: Check if port 7860 is already in use
   - **Webcam not starting**: Ensure no other applications are using the camera
   - **Attendance table not updating**: Refresh the page or restart the interface

### Performance Tips

1. **CPU Usage**: The system runs on CPU only. For better performance:
   - Close unnecessary applications
   - Use smaller camera resolution
   - Increase `DUPLICATE_THRESHOLD_HOURS` to reduce logging frequency

2. **Accuracy**: For better recognition:
   - Use multiple photos per person
   - Ensure good lighting in reference photos
   - Keep camera at appropriate distance (2-6 feet)

## Model Information

### SCRFD (Face Detection)
- **Paper**: [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714)
- **Model**: SCRFD 10G
- **Input Size**: 640x640
- **Features**: High accuracy, real-time performance

### ArcFace (Face Recognition)  
- **Paper**: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **Model**: ResNet-50 backbone
- **Embedding Size**: 512 dimensions
- **Features**: State-of-the-art accuracy, robust to variations

### FAISS (Face Search)
- **Library**: Facebook AI Similarity Search
- **Index Type**: Flat Inner Product (cosine similarity)
- **Features**: Fast similarity search, scalable

## File Structure Explanation

- **`main.py`**: Application entry point and user interface
- **`face_recognition.py`**: Core face detection and recognition logic
- **`attendance_logger.py`**: Attendance logging and Excel management
- **`config.py`**: Configuration parameters and paths
- **`models/`**: Model wrapper classes (SCRFD, ArcFace)
- **`utils/`**: Utility functions (face alignment, drawing, logging)
- **`database/`**: Face database management (FAISS integration)

## License

This project uses components from the [face-reidentification](https://github.com/yakhyo/face-reidentification) repository for face detection and recognition models.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

If you encounter any issues:

1. Check this README for troubleshooting tips
2. Verify all dependencies are installed correctly
3. Ensure model files are downloaded and placed correctly
4. Check that your camera works with other applications

For additional help, please create an issue in the repository with:
- Your operating system
- Python version
- Error messages (if any)
- Steps to reproduce the issue

---

## Quick Start Guide for Gradio Interface

### 🚀 Quick Setup (5 minutes)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download model files** to `weights/` directory:
   - `det_10g.onnx` (Face Detection)
   - `w600k_r50.onnx` (Face Recognition)

3. **Add some people** (optional - can be done via web interface):
   ```bash
   # Place photos in faces/ directory as: firstname_lastname.jpg
   faces/john_doe.jpg
   faces/jane_smith.jpg
   ```

4. **Launch Gradio interface:**
   ```bash
   python Gradio_interface.py
   ```

5. **Access web interface:**
   - Automatically opens at `http://localhost:7860`
   - Use **👤 Add Person** tab to add more people
   - Use **📹 Live Attendance** tab to start monitoring

### 🎯 Usage Tips

- **Best photo quality**: Well-lit, front-facing, clear face
- **Camera position**: 2-6 feet from subjects for optimal recognition
- **Attendance logging**: Automatic with 8-hour duplicate prevention
- **Data export**: Click "Download Attendance Excel" anytime
- **Real-time monitoring**: Table updates every 2 seconds automatically

---

**Note**: This system is designed for educational and small-scale attendance tracking purposes. For production use in larger organizations, consider additional security measures and privacy considerations.