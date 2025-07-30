from datetime import datetime


def run_financial_model(macro_data: dict, micro_data: dict) -> dict:
    """
    A mock financial model that compares macroeconomic and microeconomic data.
    """
    gdp_growth = macro_data.get("gdp_growth_rate", 0)
    inflation = macro_data.get("inflation_rate", 0)
    unemployment = macro_data.get("unemployment_rate", 0)
    nasdaq_index = micro_data.get("nasdaq_composite_index", 0)
    pe_ratio = micro_data.get("average_pe_ratio", 0)

    # A simple scoring model
    market_sentiment_score = (
        (gdp_growth * 1.5) - (inflation * 1.2) - (unemployment * 1.0)
    )
    nasdaq_health_score = (nasdaq_index / 15000) * (50 / pe_ratio)

    overall_score = (market_sentiment_score * 0.4) + (nasdaq_health_score * 0.6)

    if overall_score > 2.5:
        market_summary = "Positive market outlook. Strong economic indicators and healthy tech sector performance suggest a favorable investment climate."
    elif overall_score > 1.0:
        market_summary = "Neutral market outlook. Mixed economic signals suggest a cautious approach. Some sectors may outperform others."
    else:
        market_summary = "Negative market outlook. Weak economic indicators and poor tech sector performance suggest a high-risk investment climate."

    return {
        "report_date": datetime.now().isoformat(),
        "market_summary": market_summary,
        "model_scores": {
            "market_sentiment_score": round(market_sentiment_score, 2),
            "nasdaq_health_score": round(nasdaq_health_score, 2),
            "overall_score": round(overall_score, 2),
        },
        "raw_data": {"macro": macro_data, "micro": micro_data},
    }
