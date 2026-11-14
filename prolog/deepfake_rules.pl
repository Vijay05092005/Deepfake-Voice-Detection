% ============================================================
% DEEPFAKE VOICE DETECTION
% PROLOG KNOWLEDGE BASE
% ============================================================

% 1. AUDIO CLASSIFICATION

classify_audio(Probability, Threshold, spoof) :-
    Probability >= Threshold.

classify_audio(Probability, Threshold, bona_fide) :-
    Probability < Threshold.


% 2. CONFIDENCE LEVEL

confidence_level(Probability, high) :-
    Probability >= 0.90.

confidence_level(Probability, moderate) :-
    Probability >= 0.70,
    Probability < 0.90.

confidence_level(Probability, low) :-
    Probability < 0.70.


% 3. AUDIO ASSESSMENT

audio_assessment(spoof, high, suspicious).
audio_assessment(spoof, moderate, suspicious).
audio_assessment(spoof, low, uncertain).

audio_assessment(bona_fide, high, likely_genuine).
audio_assessment(bona_fide, moderate, requires_verification).
audio_assessment(bona_fide, low, uncertain).


% 4. RECOMMENDATIONS

recommendation(suspicious,
    'Potential synthetic or manipulated audio detected. Further verification is recommended.').

recommendation(likely_genuine,
    'Audio appears consistent with genuine speech. This result is not a guarantee of authenticity.').

recommendation(requires_verification,
    'Audio is classified as potentially genuine, but additional verification is recommended.').

recommendation(uncertain,
    'The model result is uncertain. Please perform additional audio verification.').


% 5. MAIN ANALYSIS

analyze_audio(
    SpoofProbability,
    ModelConfidence,
    Threshold,
    Result,
    Confidence,
    Assessment,
    Message
) :-

    classify_audio(
        SpoofProbability,
        Threshold,
        Result
    ),

    confidence_level(
        ModelConfidence,
        Confidence
    ),

    audio_assessment(
        Result,
        Confidence,
        Assessment
    ),

    recommendation(
        Assessment,
        Message
    ).