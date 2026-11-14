from pyswip import Prolog

# Initialize Prolog
prolog = Prolog()

# Load the knowledge base
prolog.consult(
    "Deepfake_Voice_Detection/prolog/deepfake_rules.pl"
)

print("Prolog knowledge base loaded successfully!")

# Test CNN prediction passed to Prolog
spoof_probability = 0.98
threshold = 0.95

query = list(
    prolog.query(
        f"analyze_audio({spoof_probability}, {threshold}, "
        "Result, Confidence, Assessment, Message)"
    )
)

if query:
    result = query[0]

    print("\nPROLOG ANALYSIS")
    print("----------------------")
    print("CNN Spoof Probability:", spoof_probability)
    print("Decision Threshold:", threshold)
    print("Classification:", result["Result"])
    print("Confidence:", result["Confidence"])
    print("Assessment:", result["Assessment"])
    print("Recommendation:", result["Message"])
else:
    print("No Prolog result returned.")