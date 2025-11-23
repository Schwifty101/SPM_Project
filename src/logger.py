"""
Centralized logging configuration for Loyalty AI Agent
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class AgentLogger:
    """Centralized logger for the AI Agent system"""

    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "loyalty_agent", log_file: str = "logs/agent.log") -> logging.Logger:
        """
        Get or create logger instance

        Args:
            name: Logger name
            log_file: Path to log file

        Returns:
            Configured logger instance
        """
        if cls._instance is None:
            cls._instance = cls._setup_logger(name, log_file)
        return cls._instance

    @classmethod
    def _setup_logger(cls, name: str, log_file: str) -> logging.Logger:
        """
        Configure logger with file and console handlers

        Args:
            name: Logger name
            log_file: Path to log file

        Returns:
            Configured logger
        """
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


# Convenience function
def get_logger(name: str = "loyalty_agent") -> logging.Logger:
    """Get logger instance"""
    return AgentLogger.get_logger(name)
