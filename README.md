# Football-Computer-Vision
Football computer vision project aimed to analyze players performance through a YOLO, Triplets, etc.

# Model
For football object detection I ended up choosing Yolov11n due to hardware limitations (Nvidia GTX 1660 SUPER, 6 GB VRAM).

# Datasets
The YOLO model will be fine-tuned with SoccerNet's gamestate-2024 datasets, it has player, refeere, goalkeeper and ball as classes and over 70GBs worth of football recordings.

# Preprocess
I made a subsample of Soccernet's gamestate-2024 dataset by frame striding the original recordings. This decreases the image number from 42000 to 8000, reducing redundancy between consecutive frames, trading some training diversity for feasible training time on limited hardware.

# Training
The Yolo model will be fine-tuned with said dataset, using an image size of 960 and a bach number of 6 in 100 epochs, with 20 iterations of patience if the model converges earlier. Training results are monitorized in real-time via Tensorboard, which Ultralytics has native support with.

# Libraries
Ultralytics: Provides YOLO architectures that I'll fine-tune for football object identification.

SoccerNet: Downloading SoccerNet's datasets for fine-tuning the YOLO model.

Supervision: Roboflow's library aimed to visualize our models predictions (box and classes displayings).

Albumentations: Image transformations to generate more training samples and do data augmentation, Ultralytics uses it natively while training.

PyTorch and Torchvision: Deep learning stuff and transfer learning.

OpenCV: Image handling.

XGBoost: Gradient Boosting models for analyzing player's performance.

python-dotenv: Enviroment variables handling.

