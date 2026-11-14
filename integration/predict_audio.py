import os
import torch
import numpy as np
from integration.audio_preprocessing import extract_logmel_features
from integration.load_cnn import model, DEVICE, THRESHOLD


# ==========================================
# CNN AUDIO PREDICTION
# ==========================================

def predict_audio(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # Step 1: Extract Log-Mel features
    features = extract_logmel_features(audio_path)

    # Verify feature dimensions
    if features.shape != (128, 501):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    # Step 2: Convert features to PyTorch tensor
    input_tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    # Step 3: CNN prediction
    model.eval()

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    # Step 4: Extract class probabilities
    bona_fide_probability = probabilities[0][0].item()
    spoof_probability = probabilities[0][1].item()

    # Step 5: Apply optimized threshold
    if spoof_probability >= THRESHOLD:
        prediction = "spoof"
    else:
        prediction = "bona_fide"

    # Step 6: Calculate maximum class probability
    confidence = max(
        bona_fide_probability,
        spoof_probability
    )

    return {
        "prediction": prediction,
        "bona_fide_probability": bona_fide_probability,
        "spoof_probability": spoof_probability,
        "confidence": confidence,
        "threshold": THRESHOLD
    }


# ==========================================
# TEST CNN INFERENCE
# ==========================================

if __name__ == "__main__":

    audio_path = input(
        "Enter audio file path: "
    ).strip().strip('"')

    result = predict_audio(audio_path)

    print("\n" + "=" * 45)
    print("CNN AUDIO PREDICTION RESULTS")
    print("=" * 45)

    print("Prediction:", result["prediction"])

    print(
        "Bona fide probability:",
        round(result["bona_fide_probability"], 4)
    )

    print(
        "Spoof probability:",
        round(result["spoof_probability"], 4)
    )

    print(
        "Maximum class probability:",
        round(result["confidence"], 4)
    )

    print(
        "Decision threshold:",
        result["threshold"]
    )

    print("\nCNN PREDICTION COMPLETED!")