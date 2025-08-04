# Road Segmentation with ROS and TensorFlow

## Overview
This project integrates a TensorFlow-based UNet segmentation model into a ROS node that subscribes to raw camera images, performs real-time segmentation, and publishes segmentation masks and color overlays for visualization in RViz.

## Features
- Real-time image segmentation with a UNet model.
- Publishes binary mask (`/seg/image_raw`) and colored overlay (`/seg/image_color`) ROS topics.
- Easy-to-use ROS launch file to start the pipeline.
- Visualization support in RViz.

## Installation & Setup

### Prerequisites
- ROS Noetic (tested on Ubuntu 20.04)
- Python 3.8+
- TensorFlow 2.13.x
- OpenCV, cv_bridge

### Setup Workspace

1. Clone this repository inside your ROS workspace `src` folder:

   ```bash
   cd ~/catkin_ws/src
   git clone <your-repo-url>

2. Build the workspace:
   '''bash
  cd ~/catkin_ws
  catkin_make
  source devel/setup.bash
3.  Install Python dependencies:
   pip install -r requirements.txt
Running the Project
1. Launch the segmentation pipeline and image publisher:
  roslaunch segmentation_node segmentation.launch
2. Open RViz for visualization:
   rviz
3. Add Image displays for:

/seg/image_raw (Encoding: mono8) for raw segmentation masks.

/seg/image_color (Encoding: rgb8) for colored overlays.

File Structure
segmentation_node/
├── scripts/                # ROS node scripts
│   └── segmentation_node.py
├── models/                 # Saved models
│   └── best_model.h5
├── launch/                 # Launch files
│   └── segmentation.launch
├── CMakeLists.txt
├── package.xml
train.py                   # (Optional) Training script
unet_model.py              # (Optional) Model architecture
rviz_screenshot.png        # RViz screenshot
README.md
.gitignore

Model Weights
The best_model.h5 file is included here. If the file is large, you can download it from [link_to_your_cloud_storage].

Troubleshooting
Ensure ROS environment is sourced before running nodes.

Confirm TensorFlow version matches 2.13.x for compatibility.

Check topic availability using rostopic list and rostopic echo /seg/image_raw.

License
