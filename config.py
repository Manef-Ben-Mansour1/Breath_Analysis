import os

class Config:
    # Configuration Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Dossiers
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    MODEL_FOLDER = 'models'
    
    # Extensions autorisées
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
    
    # Configuration audio
    DEFAULT_SAMPLE_RATE = 22050
    WAVELET_TYPE = 'db4'
    WAVELET_LEVELS = 2
    
    # Configuration augmentation
    DEFAULT_AUGMENTATION_COUNT = 3
    GAUSSIAN_NOISE_MIN = 0.001
    GAUSSIAN_NOISE_MAX = 0.015
    TIME_STRETCH_MIN = 0.9
    TIME_STRETCH_MAX = 1.1
    PITCH_SHIFT_MIN = -2
    PITCH_SHIFT_MAX = 2
    SHIFT_MIN = -0.5
    SHIFT_MAX = 0.5
    
    # Configuration serveur
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = os.environ.get('FLASK_ENV') == 'development'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    UPLOAD_FOLDER = 'test_uploads'
    PROCESSED_FOLDER = 'test_processed'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}