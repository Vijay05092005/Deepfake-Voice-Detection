from audio_preprocessing import extract_logmel_features

audio_path = input("Enter audio file path: ")

features = extract_logmel_features(audio_path)

print("\nAUDIO PREPROCESSING RESULTS")
print("---------------------------")

print("Feature shape:", features.shape)
print("Data type:", features.dtype)
print("Minimum value:", features.min())
print("Maximum value:", features.max())

if features.shape == (128, 501):
    print("\nSUCCESS: Feature dimensions match CNN input!")
else:
    print("\nERROR: Unexpected feature dimensions!")