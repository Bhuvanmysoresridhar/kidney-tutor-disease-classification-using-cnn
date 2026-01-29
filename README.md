# Kidney Tumor Disease Classification using CNN

A deep learning project to classify kidney tumor types using Convolutional Neural Networks (CNN). This project leverages TensorFlow/Keras for model training and provides an end-to-end pipeline for data processing, model training, evaluation, and deployment.

## Project Overview

This project aims to build an automated system for classifying kidney tumors from medical imaging data. The CNN model is trained to distinguish between different kidney tumor types, assisting medical professionals in diagnosis and treatment planning.

### Key Features

- **Deep Learning Pipeline**: Complete ML pipeline from data preprocessing to model deployment
- **CNN Architecture**: Custom CNN model optimized for medical imaging classification
- **MLflow Integration**: Experiment tracking and model versioning
- **DVC Pipeline**: Version control for datasets and pipelines
- **Web Interface**: Flask-based web application for easy model inference
- **Modular Design**: Well-organized code structure with reusable components

## Project Structure

```
.
├── config/
│   └── config.yaml              # Configuration file for model and pipeline
├── research/
│   └── trials.ipynb             # Experimental notebooks and trials
├── src/
│   └── cnnClassifier/
│       ├── components/          # Reusable pipeline components
│       ├── config/              # Configuration management
│       ├── constants/           # Project constants
│       ├── entity/              # Data models and entities
│       ├── pipeline/            # ML pipeline orchestration
│       └── utils/               # Utility functions
├── templates/
│   └── index.html               # Web UI template
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Model hyperparameters
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup configuration
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Conda or pip
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Kidney\ tumor\ disease\ classification
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n kidney python=3.8 -y
   conda activate kidney
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

## Usage

### Training the Model

1. **Update configuration** (optional)
   - Modify [config/config.yaml](config/config.yaml) for dataset paths and model parameters
   - Modify [params.yaml](params.yaml) for hyperparameters

2. **Run the DVC pipeline**
   ```bash
   dvc repro
   ```

3. **Track experiments with MLflow**
   ```bash
   mlflow ui
   ```
   Open http://localhost:5000 to view experiment metrics and models

### Running the Web Application

```bash
python app.py
```

Navigate to `http://localhost:5000` and upload kidney tumor images for classification.

### Using the Model Programmatically

```python
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

predictor = PredictionPipeline(image_path='path/to/image.jpg')
result = predictor.predict()
print(f"Predicted class: {result}")
```

## Data

The project expects medical imaging data (CT/MRI scans) for kidney tumors. Data should be organized as follows:

```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/
└── test/
    ├── class1/
    ├── class2/
    └── class3/
```

## Model Architecture

- **Framework**: TensorFlow/Keras
- **Base**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels
- **Output**: Multi-class classification (tumor types)
- **Optimization**: Adam optimizer with categorical cross-entropy loss

## Pipeline Stages

1. **Data Ingestion**: Load and validate imaging data
2. **Data Validation**: Check data quality and format
3. **Data Transformation**: Preprocessing, normalization, and augmentation
4. **Model Training**: Train CNN with specified hyperparameters
5. **Model Evaluation**: Evaluate on test set with metrics (accuracy, precision, recall, F1)
6. **Model Pushing**: Save and version model artifacts

## Results

Model performance metrics are tracked in MLflow. Key metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- Loss curves (training/validation)

See MLflow dashboard for detailed results and experiment comparisons.

## Technologies

- **TensorFlow/Keras**: Deep learning framework
- **DVC**: Data and pipeline versioning
- **MLflow**: Experiment tracking and model registry
- **Flask**: Web framework
- **NumPy/Pandas**: Data processing
- **Matplotlib/Seaborn**: Visualization
- **Python-Box**: Configuration management

## Requirements

See [requirements.txt](requirements.txt) for all dependencies:
- tensorflow==2.12.0
- pandas
- numpy
- dvc
- mlflow==2.2.2
- Flask
- Flask-Cors
- And more...

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{kidney_tumor_classification,
  title={Kidney Tumor Disease Classification using CNN},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/kidney-tumor-classification}
}
```

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

## Disclaimer

This project is for educational and research purposes. It should not be used as the sole basis for medical diagnosis. Always consult with qualified medical professionals for medical advice.