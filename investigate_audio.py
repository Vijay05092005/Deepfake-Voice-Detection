import os
import pandas as pd
from integration.predict_audio import predict_audio

folder = "test_audio"

results = []

for category in ["bonafide", "spoof"]:

    for i in range(1, 6):

        filename = f"{category}{i}.flac"
        audio_path = os.path.join(folder, filename)

        prediction = predict_audio(audio_path)

        results.append({
            "Recording": filename,
            "Actual Label": category,
            "CNN Prediction": prediction["prediction"],
            "Bona fide Probability": round(
                prediction["bona_fide_probability"] * 100, 2
            ),
            "Spoof Probability": round(
                prediction["spoof_probability"] * 100, 2
            )
        })

df = pd.DataFrame(results)

print("\nFINAL CNN TEST RESULTS")
print("=" * 80)

print(df.to_string(index=False))

correct = sum(
    df["Actual Label"] == df["CNN Prediction"]
)

print("\nCorrect Predictions:", correct, "/ 10")
print("Test Accuracy:", round(correct / 10 * 100, 2), "%")