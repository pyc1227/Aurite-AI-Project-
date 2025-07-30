"""
Database management for the AI Macro Analysis Agent.
Handles connections and operations with Neon DB.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger

from .config import DatabaseConfig


class NeonDBManager:
    """Manager for Neon DB operations."""

    def __init__(self, config: DatabaseConfig):
        """Initialize database manager with configuration."""
        self.config = config
        self.engine = None
        self.session_factory = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Neon DB."""
        try:
            self.engine = create_engine(
                self.config.url,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,
            )

            self.session_factory = sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"✅ Connected to Neon DB: {self.config.host}")

        except Exception as e:
            logger.error(f"❌ Failed to connect to Neon DB: {e}")
            raise

    def get_latest_macro_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Retrieve the latest macro economic data from the database after 2008-01-01.

        Args:
            limit: Maximum number of records to retrieve

        Returns:
            DataFrame with macro indicators (post-2008 data only)
        """
        try:
            # Handle schema-qualified table names (e.g., "macro"."macro data")
            table_name = self.config.macro_table

            # If table name contains a dot, it's schema.table format
            if "." in table_name:
                schema, table = table_name.split(".", 1)
                table_reference = f'"{schema}"."{table}"'
            else:
                # Default to public schema if no schema specified
                table_reference = f'"{table_name}"'

            query = f"""
            SELECT * FROM {table_reference}
            WHERE date >= '2008-01-01'
            ORDER BY date DESC
            LIMIT {limit}
            """

            logger.info(f"📊 Querying table: {table_reference}")
            df = pd.read_sql_query(query, self.engine)

            if df.empty:
                logger.warning(f"No macro data found in table: {table_reference}")
                return pd.DataFrame()

            logger.info(
                f"📊 Retrieved {len(df)} macro data records from {table_reference}"
            )
            return df

        except Exception as e:
            logger.error(f"❌ Error retrieving macro data from {table_reference}: {e}")
            raise

    def get_quarterly_macro_data(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve quarterly aggregated macro data.

        Args:
            start_date: Start date for data retrieval (YYYY-MM-DD format, defaults to 2008-01-01)

        Returns:
            DataFrame with quarterly macro data
        """
        try:
            # Default to post-financial crisis data
            if start_date is None:
                start_date = "2008-01-01"

            date_filter = f"WHERE date >= '{start_date}'"

            query = f"""
            SELECT 
                EXTRACT(YEAR FROM date) as year,
                EXTRACT(QUARTER FROM date) as quarter,
                AVG(vix) as vix,
                AVG(unemployment_rate) as unemployment_rate,
                AVG(fed_funds_rate) as fed_funds_rate,
                AVG(treasury_10y) as treasury_10y,
                AVG(real_gdp) as real_gdp,
                AVG(fed_balance_sheet) as fed_balance_sheet,
                COUNT(*) as record_count,
                MIN(date) as quarter_start,
                MAX(date) as quarter_end
            FROM "{self.config.macro_table}"
            {date_filter}
            GROUP BY EXTRACT(YEAR FROM date), EXTRACT(QUARTER FROM date)
            ORDER BY year DESC, quarter DESC
            """

            df = pd.read_sql_query(query, self.engine)

            if not df.empty:
                # Create quarter identifier
                df["quarter_id"] = (
                    df["year"].astype(str) + "Q" + df["quarter"].astype(str)
                )
                logger.info(f"📊 Retrieved {len(df)} quarterly records")

            return df

        except Exception as e:
            logger.error(f"❌ Error retrieving quarterly data: {e}")
            raise

    def save_prediction(
        self,
        prediction: str,
        confidence: float,
        features_used: Dict,
        model_name: str,
        target_quarter: str,
    ) -> bool:
        """
        Save prediction results to database.

        Args:
            prediction: 'bullish' or 'bearish'
            confidence: Confidence score (0-1)
            features_used: Dictionary of features and their values
            model_name: Name of the model used
            target_quarter: Target quarter (e.g., '2024Q1')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create predictions table if it doesn't exist
            self._create_predictions_table()

            prediction_data = {
                "prediction_date": datetime.now(),
                "target_quarter": target_quarter,
                "prediction": prediction.lower(),
                "confidence": confidence,
                "model_name": model_name,
                "features_json": str(features_used),  # Store as JSON string
                "created_at": datetime.now(),
            }

            # Insert prediction
            with self.engine.connect() as conn:
                insert_query = f"""
                INSERT INTO {self.config.predictions_table} 
                (prediction_date, target_quarter, prediction, confidence, model_name, features_json, created_at)
                VALUES 
                (:prediction_date, :target_quarter, :prediction, :confidence, :model_name, :features_json, :created_at)
                """

                conn.execute(text(insert_query), prediction_data)
                conn.commit()

            logger.info(
                f"💾 Saved prediction: {prediction} ({confidence:.1%}) for {target_quarter}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Error saving prediction: {e}")
            return False

    def get_prediction_history(self, limit: int = 50) -> pd.DataFrame:
        """
        Retrieve historical predictions.

        Args:
            limit: Maximum number of predictions to retrieve

        Returns:
            DataFrame with prediction history
        """
        try:
            query = f"""
            SELECT * FROM {self.config.predictions_table}
            ORDER BY prediction_date DESC
            LIMIT {limit}
            """

            df = pd.read_sql_query(query, self.engine)
            logger.info(f"📊 Retrieved {len(df)} prediction records")
            return df

        except Exception as e:
            logger.error(f"❌ Error retrieving prediction history: {e}")
            return pd.DataFrame()

    def _create_predictions_table(self) -> None:
        """Create predictions table if it doesn't exist."""
        try:
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.config.predictions_table} (
                id SERIAL PRIMARY KEY,
                prediction_date TIMESTAMP NOT NULL,
                target_quarter VARCHAR(10) NOT NULL,
                prediction VARCHAR(20) NOT NULL,
                confidence FLOAT NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                features_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            with self.engine.connect() as conn:
                conn.execute(text(create_table_query))
                conn.commit()

        except Exception as e:
            logger.error(f"❌ Error creating predictions table: {e}")
            raise

    def get_nasdaq_data(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve NASDAQ data if stored in database.

        Args:
            start_date: Start date for data retrieval

        Returns:
            DataFrame with NASDAQ data
        """
        try:
            # Check if NASDAQ table exists
            query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = 'nasdaq_data'
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchall()

            if not result:
                logger.warning("NASDAQ data table not found in database")
                return pd.DataFrame()

            date_filter = ""
            if start_date:
                date_filter = f"WHERE date >= '{start_date}'"

            nasdaq_query = f"""
            SELECT * FROM nasdaq_data
            {date_filter}
            ORDER BY date DESC
            """

            df = pd.read_sql_query(nasdaq_query, self.engine)
            logger.info(f"📊 Retrieved {len(df)} NASDAQ records")
            return df

        except Exception as e:
            logger.error(f"❌ Error retrieving NASDAQ data: {e}")
            return pd.DataFrame()

    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on database connection and tables.

        Returns:
            Dictionary with health check results
        """
        health = {"connection": False, "macro_table": False, "predictions_table": False}

        try:
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            health["connection"] = True

            # Check macro table (handle schema-qualified names)
            table_name = self.config.macro_table
            if "." in table_name:
                schema, table = table_name.split(".", 1)
                table_reference = f'"{schema}"."{table}"'
            else:
                table_reference = f'"{table_name}"'

            query = f"""
            SELECT COUNT(*) FROM {table_reference}
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).scalar()
                health["macro_table"] = result is not None

            # Check predictions table (create if doesn't exist)
            self._create_predictions_table()
            health["predictions_table"] = True

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")

        return health

    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("🔌 Database connection closed")
