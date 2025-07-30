"""
Main AI Macro Analysis Agent.
Orchestrates all components to provide quarterly NASDAQ 100 predictions.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from .config import Config
from .database import NeonDBManager
from .feature_engineer import FeatureEngineer
from .model_manager import ModelManager
from .openai_client import OpenAIClient


class MacroAnalysisAgent:
    """AI Agent for quarterly macro analysis and NASDAQ 100 predictions."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI agent with all components.

        Args:
            config_path: Optional path to configuration file
        """
        try:
            # Load configuration
            self.config = Config()

            # Initialize components
            self.db_manager = NeonDBManager(self.config.database)
            self.feature_engineer = FeatureEngineer()
            self.model_manager = ModelManager(self.config.model)
            self.openai_client = OpenAIClient(self.config.openai)

            # Agent state
            self.is_initialized = False
            self.last_prediction = None
            self.last_features = None

            logger.info(f"🤖 {self.config.agent.name} initialized successfully")

        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise

    def initialize(self) -> bool:
        """
        Initialize the agent by loading models and validating connections.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing MacroAnalysisAgent...")

            # Load trained model first
            if not self.model_manager.load_model():
                logger.error("❌ Failed to load trained model")
                return False

            # Perform health checks after model is loaded
            health = self.health_check()

            if not all(health.values()):
                logger.error(f"❌ Health check failed: {health}")
                return False

            self.is_initialized = True
            logger.info("✅ Agent initialization complete!")

            return True

        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            return False

    def predict_next_quarter(
        self, include_openai_analysis: bool = True, market_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate prediction for the next quarter.

        Args:
            include_openai_analysis: Whether to include OpenAI-generated analysis
            market_context: Optional additional market context

        Returns:
            Complete prediction results with analysis
        """
        try:
            if not self.is_initialized:
                raise ValueError("Agent not initialized. Call initialize() first.")

            logger.info("🎯 Generating next quarter prediction...")

            # Step 1: Retrieve latest macro data
            macro_data = self.db_manager.get_latest_macro_data()
            if macro_data is None or macro_data.empty:
                raise ValueError("No macro data available in database")

            # Step 2: Engineer features
            featured_data = self.feature_engineer.create_quarterly_features(macro_data)
            latest_features = self.feature_engineer.get_latest_quarter_features()

            if not latest_features:
                raise ValueError("Failed to extract features from data")

            # Step 3: Make prediction
            prediction_result = self.model_manager.predict(latest_features)

            # Step 4: Get feature importance
            feature_importance = self.model_manager.get_feature_importance(15)

            # Step 5: Generate target quarter identifier
            current_date = datetime.now()
            next_quarter = self._get_next_quarter(current_date)

            # Step 6: Save prediction to database
            self.db_manager.save_prediction(
                prediction=prediction_result["prediction"],
                confidence=prediction_result["confidence"],
                features_used=latest_features,
                model_name=prediction_result["model_name"],
                target_quarter=next_quarter,
            )

            # Step 7: Generate OpenAI analysis (if requested)
            openai_analysis = {}
            if include_openai_analysis:
                try:
                    # Generate comprehensive report
                    prediction_report = self.openai_client.generate_prediction_report(
                        prediction_result, latest_features, market_context
                    )

                    # Generate factor explanation
                    factor_explanation = self.openai_client.explain_prediction_factors(
                        prediction_result, feature_importance or [], latest_features
                    )

                    # Generate market summary
                    market_summary = self.openai_client.generate_market_summary(
                        latest_features, self._get_recent_predictions(5)
                    )

                    openai_analysis = {
                        "prediction_report": prediction_report,
                        "factor_explanation": factor_explanation,
                        "market_summary": market_summary,
                    }

                except Exception as e:
                    logger.warning(f"⚠️ OpenAI analysis failed: {e}")
                    openai_analysis = {"error": str(e)}

            # Step 8: Compile complete results
            complete_result = {
                "prediction": prediction_result,
                "target_quarter": next_quarter,
                "features": latest_features,
                "feature_importance": feature_importance,
                "data_summary": {
                    "macro_records": len(macro_data),
                    "quarterly_periods": len(featured_data),
                    "features_engineered": len(
                        self.feature_engineer.get_feature_names()
                    ),
                },
                "openai_analysis": openai_analysis,
                "agent_info": {
                    "name": self.config.agent.name,
                    "timestamp": datetime.now().isoformat(),
                    "model_info": self.model_manager.get_model_info(),
                },
            }

            # Store for reference
            self.last_prediction = complete_result
            self.last_features = latest_features

            logger.info(
                f"✅ Prediction complete: {prediction_result['prediction'].upper()} "
                f"({prediction_result['confidence']:.1%})"
            )

            return complete_result

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

    def get_prediction_summary(self, include_technical: bool = True) -> str:
        """
        Get a formatted summary of the latest prediction.

        Args:
            include_technical: Whether to include technical details

        Returns:
            Formatted prediction summary
        """
        if not self.last_prediction:
            return "No predictions available. Run predict_next_quarter() first."

        try:
            result = self.last_prediction
            prediction = result["prediction"]

            # Basic summary
            summary = f"""
🤖 NASDAQ 100 QUARTERLY PREDICTION
{"=" * 50}

🎯 PREDICTION: {prediction["prediction"].upper()}
📊 CONFIDENCE: {prediction["confidence"]:.1%}
📅 TARGET QUARTER: {result["target_quarter"]}
🕐 GENERATED: {result["agent_info"]["timestamp"][:19]}

📈 PROBABILITY BREAKDOWN:
   • Bullish: {prediction["probabilities"]["bullish"]:.1%}
   • Bearish: {prediction["probabilities"]["bearish"]:.1%}

🤖 MODEL INFO:
   • Type: {prediction["model_name"]}
   • Features Used: {prediction["features_used"]}
   • Confidence Threshold: {prediction["threshold"]:.1%}
   • Meets Threshold: {"✅ YES" if prediction["meets_threshold"] else "❌ NO"}
            """

            # Add enhanced feature importance analysis
            if include_technical and result.get("feature_importance"):
                summary += "\n🔍 TOP INFLUENTIAL FEATURES:"

                # Categorize features for better insight
                feature_categories = {
                    "lag": [],
                    "trend": [],
                    "ar": [],
                    "cycle": [],
                    "cross": [],
                    "diff": [],
                    "zscore": [],
                    "base": [],
                }

                for i, (feature, importance) in enumerate(
                    result["feature_importance"][:15], 1
                ):
                    value = result["features"].get(feature, 0.0)

                    # Categorize the feature
                    if "_lag_" in feature or "_lag1_" in feature or "_lag2_" in feature:
                        feature_categories["lag"].append((feature, importance, value))
                    elif (
                        "_trend_" in feature
                        or "_slope_" in feature
                        or "_acceleration" in feature
                    ):
                        feature_categories["trend"].append((feature, importance, value))
                    elif (
                        "_ar1" in feature
                        or "_mean_reversion" in feature
                        or "_persistence" in feature
                    ):
                        feature_categories["ar"].append((feature, importance, value))
                    elif "cycle" in feature or "recession" in feature:
                        feature_categories["cycle"].append((feature, importance, value))
                    elif "_x_" in feature or "_agreement" in feature:
                        feature_categories["cross"].append((feature, importance, value))
                    elif "_diff_" in feature or "_pct_" in feature:
                        feature_categories["diff"].append((feature, importance, value))
                    elif "_zscore" in feature or "_regime" in feature:
                        feature_categories["zscore"].append(
                            (feature, importance, value)
                        )
                    else:
                        feature_categories["base"].append((feature, importance, value))

                    summary += f"\n   {i:2d}. {feature}: {value:.3f} (importance: {importance:.3f})"

                # Add feature category summary
                if any(feature_categories.values()):
                    summary += "\n\n🎯 FEATURE CATEGORY ANALYSIS:"
                    if feature_categories["lag"]:
                        summary += f"\n   📅 Lag Features: {len(feature_categories['lag'])} influential (previous quarter effects)"
                    if feature_categories["trend"]:
                        summary += f"\n   📈 Trend Features: {len(feature_categories['trend'])} influential (long-term patterns)"
                    if feature_categories["ar"]:
                        summary += f"\n   🔄 Autoregressive: {len(feature_categories['ar'])} influential (temporal dependencies)"
                    if feature_categories["cycle"]:
                        summary += f"\n   🌊 Cyclical Features: {len(feature_categories['cycle'])} influential (business cycles)"
                    if feature_categories["cross"]:
                        summary += f"\n   🔗 Cross-lag Features: {len(feature_categories['cross'])} influential (economic relationships)"

            # Add OpenAI analysis if available
            if (
                result.get("openai_analysis")
                and "prediction_report" in result["openai_analysis"]
            ):
                summary += f"\n\n📝 AI ANALYSIS:\n{result['openai_analysis']['prediction_report']}"

            return summary.strip()

        except Exception as e:
            logger.error(f"❌ Error creating summary: {e}")
            return f"Error creating summary: {e}"

    def get_market_analysis(self) -> str:
        """
        Get detailed market analysis based on current conditions.

        Returns:
            Market analysis report
        """
        try:
            if not self.last_features:
                # Get latest data
                macro_data = self.db_manager.get_latest_macro_data(50)
                if macro_data.empty:
                    return "No macro data available for analysis."

                # Extract key indicators
                latest_row = macro_data.iloc[0]
                key_indicators = {}
                for col in [
                    "vix",
                    "unemployment_rate",
                    "fed_funds_rate",
                    "treasury_10y",
                    "real_gdp",
                ]:
                    if col in latest_row:
                        key_indicators[col] = latest_row[col]
            else:
                key_indicators = self.last_features

            # Generate market summary
            market_summary = self.openai_client.generate_market_summary(
                key_indicators, self._get_recent_predictions(10)
            )

            return market_summary

        except Exception as e:
            logger.error(f"❌ Market analysis failed: {e}")
            return f"Market analysis unavailable: {e}"

    def retrain_model(self, save_model: bool = True) -> bool:
        """
        Retrain the model with latest data.

        Args:
            save_model: Whether to save the trained model

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🔄 Starting model retraining...")

            # Get historical data
            macro_data = self.db_manager.get_latest_macro_data(limit=1000)
            if len(macro_data) < 50:
                logger.error("❌ Insufficient data for training (need >50 records)")
                return False

            # Engineer features
            featured_data = self.feature_engineer.create_quarterly_features(macro_data)

            if len(featured_data) < 20:
                logger.error(
                    "❌ Insufficient quarterly data for training (need >20 quarters)"
                )
                return False

            # This would implement the full training pipeline from your notebook
            # For now, return success if data is available
            logger.info(f"✅ Training data prepared: {len(featured_data)} quarters")

            # TODO: Implement full training pipeline
            logger.warning("⚠️ Full retraining pipeline not implemented yet")

            return True

        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
            return False

    def health_check(self) -> Dict[str, bool]:
        """
        Perform comprehensive health check of all components.

        Returns:
            Dictionary with health status of each component
        """
        health = {}

        try:
            # Database health
            db_health = self.db_manager.health_check()
            health["database"] = db_health.get("connection", False)
            health["macro_data"] = db_health.get("macro_table", False)

            # Model health
            model_info = self.model_manager.get_model_info()
            health["model_loaded"] = model_info["loaded"]
            health["scaler_available"] = model_info["scaler_fitted"]

            # OpenAI health (basic check)
            try:
                # Simple test to check if API key works
                health["openai"] = len(self.config.openai.api_key) > 10
            except:
                health["openai"] = False

            # Configuration validation
            health["config"] = self.config.validate()

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            health["error"] = str(e)

        return health

    def _get_next_quarter(self, current_date: datetime) -> str:
        """Get the next quarter identifier."""
        current_quarter = (current_date.month - 1) // 3 + 1
        if current_quarter == 4:
            next_year = current_date.year + 1
            next_quarter = 1
        else:
            next_year = current_date.year
            next_quarter = current_quarter + 1

        return f"{next_year}Q{next_quarter}"

    def _get_recent_predictions(self, limit: int = 5) -> List[Dict]:
        """Get recent prediction history."""
        try:
            predictions_df = self.db_manager.get_prediction_history(limit)
            if predictions_df.empty:
                return []

            return predictions_df.to_dict("records")
        except:
            return []

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status information.

        Returns:
            Dictionary with agent status and statistics
        """
        try:
            health = self.health_check()

            status = {
                "agent_name": self.config.agent.name,
                "initialized": self.is_initialized,
                "health": health,
                "last_prediction": {
                    "available": self.last_prediction is not None,
                    "timestamp": self.last_prediction["agent_info"]["timestamp"]
                    if self.last_prediction
                    else None,
                    "direction": self.last_prediction["prediction"]["prediction"]
                    if self.last_prediction
                    else None,
                    "confidence": self.last_prediction["prediction"]["confidence"]
                    if self.last_prediction
                    else None,
                },
                "model_info": self.model_manager.get_model_info(),
                "data_status": {
                    "macro_records": len(self.db_manager.get_latest_macro_data(10)),
                    "feature_count": len(self.feature_engineer.get_feature_names()),
                },
                "configuration": {
                    "confidence_threshold": self.config.model.confidence_threshold,
                    "openai_model": self.config.openai.model,
                    "log_level": self.config.agent.log_level,
                },
            }

            return status

        except Exception as e:
            logger.error(f"❌ Error getting agent status: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """Clean up resources and close connections."""
        try:
            if hasattr(self, "db_manager"):
                self.db_manager.close()

            logger.info("🔌 Agent resources cleaned up")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
