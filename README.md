🧱 Step 1 — Install Packages (Terminal)
Create and activate a virtual environment, then install dependencies:
python3 -m venv spinach_env
source spinach_env/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio # MPS version installs automatically on Apple Silicon (M1/M2)
pip install -r requirements.txt
📥 Step 2 — Download and Place the Datasets
Dataset 1
Go to:
https://data.mendeley.com/datasets/sy69db2nz5/2
Click "Download All"
Extract the ZIP file into:
Spinach_Project/data/raw/dataset1/
Dataset 2
Go to:
https://data.mendeley.com/datasets/fzgghkgf6g/1
Click "Download All"
Extract the ZIP file into:
Spinach_Project/data/raw/dataset2/
After this step, your structure should look like:
data/
├── raw/
│ ├── dataset1/
│ └── dataset2/
⚙️ Step 3 — Prepare & Split the Dataset (Terminal)
Run the dataset preparation script:
python 01_prepare_dataset.py
This script will:
Read images from data/raw/
Convert annotations to YOLO format (if required)
Split data into train / val / test
Save processed data into:
data/splits/
📓 Step 4 — Open the Notebook
Launch Jupyter Notebook:
jupyter notebook spinach_detection.ipynb
From there, you can:
Train the model
Evaluate performance
Generate metrics and visualizations
📂 Final Expected Folder Structure
After completing all steps:
Spinach_Project/
│
├── data/
│ ├── raw/
│ │ ├── dataset1/
│ │ └── dataset2/
│ │
│ └── splits/
│ ├── train/
│ ├── val/
│ └── test/
│
├── 01_prepare_dataset.py
├── spinach_detection.ipynb
├── dataset.yaml
└── requirements.txt
