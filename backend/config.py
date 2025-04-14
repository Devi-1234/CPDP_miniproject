import os

class Config:
    DEBUG = True  # Enable Flask debug mode
    SECRET_KEY = os.environ.get("SECRET_KEY", "a2d654dd13ee970bb16b3e4808d38093299771376b905f806466a11907eec198")
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = {"csv"}  # Only allow CSV file uploads

    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

class DevelopmentConfig(Config):
    ENV = "development"
    DEBUG = True

class ProductionConfig(Config):
    ENV = "production"
    DEBUG = False
