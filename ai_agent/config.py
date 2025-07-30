"""
Configuration management for the AI Macro Analysis Agent.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    url: str = Field(..., description="Complete database URL")
    host: str = Field(..., description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    macro_table: str = Field(
        default="macro indicators", description="Macro data table name"
    )
    predictions_table: str = Field(
        default="market_predictions", description="Predictions table name"
    )


class OpenAIConfig(BaseModel):
    """OpenAI API configuration settings."""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="OpenAI model to use")
    max_tokens: int = Field(default=1000, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, description="Response creativity level")


class ModelConfig(BaseModel):
    """Machine learning model configuration."""

    model_path: str = Field(
        default="models/enhanced_nasdaq_model.pkl", description="Path to saved model"
    )
    scaler_path: str = Field(
        default="models/feature_scaler.pkl", description="Path to saved scaler"
    )
    feature_columns_path: str = Field(
        default="models/feature_columns.json", description="Path to feature columns"
    )
    confidence_threshold: float = Field(
        default=0.6, description="Minimum confidence for predictions"
    )


class AgentConfig(BaseModel):
    """General agent configuration."""

    name: str = Field(default="MacroAnalysisAgent", description="Agent name")
    log_level: str = Field(default="INFO", description="Logging level")
    batch_size: int = Field(default=100, description="Batch size for processing")


class Config:
    """Main configuration class for the AI agent."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Get database URL - try different common environment variable names
        db_url = None
        for env_var in ["NEON_DATABASE_URL", "NEON_DB_URL", "DATABASE_URL"]:
            db_url = os.getenv(env_var)
            if db_url:
                break

        if not db_url:
            raise ValueError(
                "Database URL not found. Please set NEON_DATABASE_URL, NEON_DB_URL, or DATABASE_URL"
            )

        # Parse URL for individual components (for backward compatibility)
        import urllib.parse

        parsed = urllib.parse.urlparse(db_url)

        self.database = DatabaseConfig(
            url=db_url,
            host=parsed.hostname or self._get_env("NEON_DB_HOST", "localhost"),
            port=parsed.port or int(self._get_env("NEON_DB_PORT", "5432")),
            name=parsed.path.lstrip("/") or self._get_env("NEON_DB_NAME", "postgres"),
            user=parsed.username or self._get_env("NEON_DB_USER", "postgres"),
            password=parsed.password or self._get_env("NEON_DB_PASSWORD", ""),
            macro_table=self._get_env(
                "MACRO_TABLE_NAME", "macro_indicators"
            ),  # Supports schema.table format
            predictions_table=self._get_env(
                "PREDICTIONS_TABLE_NAME", "market_predictions"
            ),
        )

        # OpenAI configuration (optional)
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai = OpenAIConfig(
            api_key=openai_api_key,
            model=self._get_env("OPENAI_MODEL", "gpt-4"),
            max_tokens=int(self._get_env("OPENAI_MAX_TOKENS", "1000")),
            temperature=float(self._get_env("OPENAI_TEMPERATURE", "0.7")),
        )

        self.model = ModelConfig(
            model_path=self._get_env("MODEL_PATH", "models/enhanced_nasdaq_model.pkl"),
            scaler_path=self._get_env("SCALER_PATH", "models/feature_scaler.pkl"),
            feature_columns_path=self._get_env(
                "FEATURE_COLUMNS_PATH", "models/feature_columns.json"
            ),
            confidence_threshold=float(
                self._get_env("PREDICTION_CONFIDENCE_THRESHOLD", "0.6")
            ),
        )

        self.agent = AgentConfig(
            name=self._get_env("AGENT_NAME", "MacroAnalysisAgent"),
            log_level=self._get_env("LOG_LEVEL", "INFO"),
            batch_size=int(self._get_env("BATCH_SIZE", "100")),
        )

    @staticmethod
    def _get_env(key: str, default: Optional[str] = None) -> str:
        """Get environment variable with optional default."""
        value = os.getenv(key, default)
        if value is None and default is None:
            raise ValueError(f"Environment variable {key} is required but not set")
        return value or ""

    def validate(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Test database connection parameters
            if not all([self.database.host, self.database.name, self.database.user]):
                return False

            # Test OpenAI API key
            if not self.openai.api_key or not self.openai.api_key.startswith("sk-"):
                return False

            # Test model paths exist (if they should)
            # Note: We'll create these during model training

            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation of config (hiding sensitive data)."""
        return f"""
Config(
    database=DatabaseConfig(host='{self.database.host}', name='{self.database.name}'),
    openai=OpenAIConfig(model='{self.openai.model}'),
    model=ModelConfig(confidence_threshold={self.model.confidence_threshold}),
    agent=AgentConfig(name='{self.agent.name}')
)
        """.strip()
