# Domain Adaptation for Machine Learning Models

This project implements domain adaptation techniques for machine learning models, focusing on CORAL (Correlation Alignment) and TCA (Transfer Component Analysis) methods.

## Project Structure

```
.
├── backend/                 # Flask backend server
│   ├── app.py              # Main Flask application
│   ├── models/             # Machine learning models
│   │   ├── coral_model.py  # CORAL implementation
│   │   └── tca_coral_model.py  # TCA + CORAL implementation
│   ├── uploads/            # Uploaded files
│   └── datasets/           # Dataset files
└── frontend/               # React frontend
    ├── public/             # Static files
    └── src/                # React source code
        ├── components/     # React components
        ├── pages/          # Page components
        └── App.js          # Main App component
```

## Features

- CORAL (Correlation Alignment) for domain adaptation
- Support for multiple machine learning models:
  - Random Forest
  - Logistic Regression
  - XGBoost
- Feature selection and preprocessing
- Model evaluation metrics
- Web interface for model training and evaluation

## Setup

### Backend

1. Create and activate virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask server:
```bash
python app.py
```

### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

## Usage

1. Access the web interface at `http://localhost:3000`
2. Upload source and target datasets
3. Select model adaptation method (CORAL)
4. Choose machine learning model
5. Train and evaluate the model

## License

MIT License 