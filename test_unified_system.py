#!/usr/bin/env python3
"""
Test Unified System
==================
Test that both example_usage.py and realistic_prediction.py work with the unified model.
"""

import sys
import subprocess
from pathlib import Path


def test_realistic_prediction():
    """Test realistic_prediction.py."""
    print("🧪 Testing realistic_prediction.py...")
    print("-" * 40)

    try:
        result = subprocess.run(
            [sys.executable, "realistic_prediction.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ realistic_prediction.py: SUCCESS")
            # Extract key information
            output = result.stdout
            if "PREDICTION:" in output and "CONFIDENCE:" in output:
                lines = output.split("\n")
                for line in lines:
                    if "PREDICTION:" in line:
                        print(f"   📈 {line.strip()}")
                    elif "CONFIDENCE:" in line:
                        print(f"   📊 {line.strip()}")
            return True
        else:
            print("❌ realistic_prediction.py: FAILED")
            print(f"   Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ realistic_prediction.py: ERROR - {e}")
        return False


def test_example_usage():
    """Test example_usage.py with option 2."""
    print("\n🧪 Testing example_usage.py (option 2)...")
    print("-" * 40)

    try:
        # Use subprocess with input
        result = subprocess.run(
            [sys.executable, "example_usage.py"],
            input="2\n",
            text=True,
            capture_output=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ example_usage.py: SUCCESS")
            # Check if it loaded the unified model
            output = result.stdout
            if "Unified Logistic Regression" in output:
                print("   🤖 Loaded unified model successfully")
            if "QUICK PREDICTION DEMO" in output:
                print("   🚀 Quick prediction demo executed")
            return True
        else:
            print("❌ example_usage.py: FAILED")
            print(f"   Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ example_usage.py: ERROR - {e}")
        return False


def check_model_files():
    """Check that unified model files exist."""
    print("🔍 Checking unified model files...")
    print("-" * 40)

    model_files = [
        "models/enhanced_nasdaq_model.pkl",
        "models/feature_scaler.pkl",
        "models/feature_columns.json",
        "models/model_metadata.json",
    ]

    all_exist = True
    for file_path in model_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("🚀 UNIFIED SYSTEM TEST")
    print("=" * 50)
    print("Testing compatibility between systems...")
    print("")

    # Check model files
    model_ok = check_model_files()
    print("")

    if not model_ok:
        print("❌ Model files missing. Run train_unified_model.py first.")
        return False

    # Test realistic prediction
    realistic_ok = test_realistic_prediction()

    # Test example usage
    example_ok = test_example_usage()

    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Model Files: {'✅' if model_ok else '❌'}")
    print(f"realistic_prediction.py: {'✅' if realistic_ok else '❌'}")
    print(f"example_usage.py: {'✅' if example_ok else '❌'}")

    if model_ok and realistic_ok and example_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Unified system is working correctly")
        print("✅ Both scripts work with the same model")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Check the errors above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
