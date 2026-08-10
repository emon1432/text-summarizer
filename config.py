"""Configuration settings for the AI Text Summarization Web Application.

This module houses enterprise architectural settings, model hyperparameters,
and environmental fallback parameters. All runtime constants are encapsulated
within dedicated classes for clean injection into the Flask server.
"""

import os
import secrets
from typing import Optional


class Config:
    """Base configuration class encapsulating application settings and hyperparameters.
    
    Attributes:
        SECRET_KEY (str): Cryptographic secret for session security and CSRF protection.
        UPLOAD_FOLDER (str): Absolute or relative storage directory for incoming files.
        MAX_CONTENT_LENGTH (int): Maximum permitted payload size (in bytes).
        MODEL_NAME (str): Hugging Face repository hub identifier for transformer architecture.
        MAX_INPUT_WORDS (int): Upper bound limit for input word submission in web UI.
        TOKEN_CHUNK_LIMIT (int): Maximum token window supported natively by BART architecture.
        DEFAULT_MAX_LENGTH (int): Default upper bound on token output length for summarizer.
        DEFAULT_MIN_LENGTH (int): Default lower bound on token output length for summarizer.
        DEFAULT_NUM_BEAMS (int): Number of beams utilized during Beam Search decoding.
        LENGTH_PENALTY (float): Exponential penalty modifier applied to sequence lengths.
        DEVICE (Optional[str]): Computing device target override ('cuda', 'cpu', or None for auto).
    """

    # Flask Core Configurations
    SECRET_KEY: str = os.getenv('SECRET_KEY', secrets.token_hex(32))
    BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER: str = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16 MB max request size

    # Transformer Model Configuration
    MODEL_NAME: str = 'facebook/bart-large-cnn'
    MAX_INPUT_WORDS: int = 10000
    TOKEN_CHUNK_LIMIT: int = 1024  # Maximum context tokens supported by BART

    # Summarization Inference Hyperparameters
    DEFAULT_MAX_LENGTH: int = 130
    DEFAULT_MIN_LENGTH: int = 30
    DEFAULT_NUM_BEAMS: int = 4
    LENGTH_PENALTY: float = 2.0

    # Device allocation: When None, model module automatically detects CUDA presence
    DEVICE: Optional[str] = os.getenv('INFERENCE_DEVICE', None)

    @classmethod
    def initialize_app(cls, app=None) -> None:
        """Performs initialization routines required for application startup.

        Args:
            app (Optional[flask.Flask]): Target Flask application instance.
        """
        if not os.path.exists(cls.UPLOAD_FOLDER):
            os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)


class DevelopmentConfig(Config):
    """Configuration overlay specialized for local interactive development."""
    DEBUG: bool = True
    TESTING: bool = False


class ProductionConfig(Config):
    """Configuration overlay specialized for hardened enterprise deployment."""
    DEBUG: bool = False
    TESTING: bool = False


# Central environment selection mapping
config_by_name: dict[str, type[Config]] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
