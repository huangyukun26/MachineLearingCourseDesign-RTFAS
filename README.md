# Real-Time Face Anti-Spoofing System (RTFAS)

This project implements a real-time face anti-spoofing system using a dual-stream autoencoder architecture. The system can effectively detect various types of presentation attacks, including printed photos, digital displays, and video replays.

## Features

- Dual-stream autoencoder with RGB and depth information processing
- Attention mechanism for better feature extraction
- Support for multiple types of presentation attacks
- Real-time detection capability
- Comprehensive evaluation metrics
- Easy-to-use training and evaluation scripts

## Project Structure

```
.
├── models/                     # Model architecture and weights
│   ├── autoencoder_model.py   # Base autoencoder implementation
│   ├── dual_stream_autoencoder.py  # Improved dual-stream model
│   └── models_train_pic/      # Training visualization
├── data/                      # Data processing and storage
│   ├── process_personal_data.py    # Personal data processing
│   ├── process_nuaa.py       # NUAA dataset processing
│   └── process_test_nuaa.py  # Test data processing
├── evaluation_results/        # Evaluation outputs
├── analysis_results/         # Analysis outputs and visualizations
├── train_dual_stream.py      # Training script
└── evaluate_model.py         # Evaluation script
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Conda package manager (recommended)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/huangyukun26/MachineLearingCourseDesign-RTFAS.git
   cd MachineLearingCourseDesign-RTFAS
   ```

2. Set up the environment:

   Option 1: Using Conda (Recommended)

   ```bash
   # Create a new conda environment
   conda create -n RTFAS python=3.8
   conda activate RTFAS

   # Install PyTorch with CUDA support
   conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

   # Install other dependencies
   conda install -y numpy==1.24.4
   conda install -y opencv-python==4.5.3.56
   conda install -y pillow==10.2.0
   conda install -y matplotlib==3.7.5
   conda install -y scikit-learn==1.3.2
   conda install -y seaborn==0.13.2
   conda install -y pandas==2.0.3
   conda install -y tqdm==4.66.5
   pip install qqdm==0.0.7
   ```

   Option 2: Using pip and venv

   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows

   # Install dependencies
   pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121
   pip install numpy==1.24.4
   pip install opencv-python==4.5.3.56
   pip install pillow==10.2.0
   pip install matplotlib==3.7.5
   pip install scikit-learn==1.3.2
   pip install seaborn==0.13.2
   pip install pandas==2.0.3
   pip install tqdm==4.66.5
   pip install qqdm==0.0.7
   ```

3. Verify the installation:

   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. Download required datasets:
   - NUAA Imposter Dataset (required for testing):
     ```bash
     # Create data directories
     mkdir -p data/public_datasets/NUAA
     # Download and extract NUAA dataset (73M version with face detection)
     # Place in data/public_datasets/NUAA/
     ```

## Dataset Preparation

**If you need datasets for this course design, please contact me huangyukun@bjfu.edu.cn**

1. Prepare the NUAA dataset:

   ```bash
   python data/process_nuaa.py
   ```

2. (Optional) Add personal data:
   - Place your images in `data/personalData/` following the structure:
     ```
     data/personalData/
     ├── real/
     │   ├── normal/
     │   └── expressions/
     └── fake/
         ├── printed/
         └── digital/
     ```
   - Process personal data:
     ```bash
     python data/process_personal_data.py
     ```

## Training

1. Train the dual-stream model:
   ```bash
   python train_dual_stream.py
   ```

The training process will:

- Save the best model to `models/best_dual_stream_model.pt`
- Generate training loss plots in `models/models_train_pic/`
- Use GPU if available, fallback to CPU if not

## Evaluation

Run the evaluation script:

```bash
python evaluate_model.py
```

This will generate:

- ROC curves
- Confusion matrices
- Error distribution plots
- Performance metrics

Results will be saved in `evaluation_results/`.

## Performance

On the NUAA test set:

- AUC: 0.9876
- Accuracy: 0.9234
- Precision: 0.9156
- Recall: 0.9312
- F1-score: 0.9234

## Memory Requirements

- Minimum 8GB RAM
- GPU with 6GB VRAM recommended
- For systems with limited GPU memory, adjust batch size in training and evaluation scripts

## Troubleshooting

If you encounter GPU out of memory errors:

1. Reduce batch size in `evaluate_model.py`
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. Use CPU for evaluation if GPU memory is insufficient

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NUAA Imposter Database
- Face Liveness Detection Challenge
- PyTorch Team

## Contact

For any questions or issues, please open an issue on GitHub or contact through [GitHub Issues](https://github.com/huangyukun26/MachineLearingCourseDesign-RTFAS/issues).
