import os
from pathlib import Path
from pyswip import Prolog

from predict_audio import predict_audio


# ============================================================
# CNN - PROLOG INTEGRATION
# ============================================================

# Locate the Prolog knowledge base
CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CURRENT_DIR.parent

PROLOG_FILE = (
    PROJECT_DIR / "prolog" / "deepfake_rules.pl"
)


# ============================================================
# INITIALIZE PROLOG
# ============================================================

prolog = Prolog()

prolog.consult(PROLOG_FILE.as_posix())

print("Prolog knowledge base loaded successfully!")


# ============================================================
# CNN AND PROLOG ANALYSIS
# ============================================================

def analyze_audio_with_prolog(audio_path):

    # Step 1: Obtain CNN prediction
    cnn_result = predict_audio(audio_path)

    spoof_probability = cnn_result[
        "spoof_probability"
    ]

    threshold = cnn_result["threshold"]

    # Step 2: Pass CNN output to Prolog
    model_confidence = cnn_result["confidence"]

    query = (
        f"analyze_audio("
        f"{spoof_probability:.10f}, "
        f"{model_confidence:.10f}, "
        f"{threshold:.10f}, "
        f"Result, Confidence, Assessment, Message)"
    )

    results = list(prolog.query(query))

    if not results:
        raise RuntimeError(
            "Prolog could not generate an assessment."
        )

    result = results[0]

    # Step 3: Combine CNN and Prolog results
    final_result = {
        "cnn_prediction": cnn_result["prediction"],
        "bona_fide_probability":
            cnn_result["bona_fide_probability"],
        "spoof_probability": spoof_probability,
        "threshold": threshold,
        "prolog_classification": result["Result"],
        "confidence": result["Confidence"],
        "assessment": result["Assessment"],
        "recommendation": result["Message"]
    }

    return final_result


# ============================================================
# TEST COMPLETE INTEGRATION
# ============================================================

if __name__ == "__main__":

    audio_path = input(
        "Enter audio file path: "
    ).strip().strip('"')

    final_result = analyze_audio_with_prolog(
        audio_path
    )

    print("\n" + "=" * 55)
    print("DEEPFAKE VOICE DETECTION - FINAL ANALYSIS")
    print("=" * 55)

    print("\nCNN RESULTS")
    print("-" * 30)

    print(
        "CNN Classification:",
        final_result["cnn_prediction"]
    )

    print(
        "Bona fide Probability:",
        round(final_result["bona_fide_probability"], 4)
    )

    print(
        "Spoof Probability:",
        round(final_result["spoof_probability"], 4)
    )

    print(
        "Decision Threshold:",
        final_result["threshold"]
    )

    print("\nPROLOG REASONING")
    print("-" * 30)

    print(
        "Classification:",
        final_result["prolog_classification"]
    )

    print(
        "Confidence Level:",
        final_result["confidence"]
    )

    print(
        "Audio Assessment:",
        final_result["assessment"]
    )

    print(
        "Recommendation:",
        final_result["recommendation"]
    )

    print("\n" + "=" * 55)
    print("CNN-PROLOG INTEGRATION COMPLETED!")
    print("=" * 55)