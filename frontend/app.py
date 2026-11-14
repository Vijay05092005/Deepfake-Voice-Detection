# ============================================================
# DEEPFAKE VOICE DETECTION SYSTEM
# Streamlit Frontend - CNN + Prolog Integration
# ============================================================

import streamlit as st
import sys
import tempfile
from pathlib import Path


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
INTEGRATION_DIR = PROJECT_DIR / "integration"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(INTEGRATION_DIR))


# ============================================================
# IMPORT CNN-PROLOG INTEGRATION
# ============================================================

from integration.cnn_prolog_integration import analyze_audio_with_prolog


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Voice Detection System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9CA3AF;
    margin-bottom: 25px;
}

.section-heading {
    font-size: 25px;
    font-weight: 650;
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-card {
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
}

.genuine-card {
    background-color: rgba(22, 163, 74, 0.15);
    border: 1px solid #16A34A;
}

.spoof-card {
    background-color: rgba(220, 38, 38, 0.15);
    border: 1px solid #DC2626;
}

.result-label {
    font-size: 30px;
    font-weight: 700;
}

.small-note {
    color: #9CA3AF;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #9CA3AF;
    padding: 15px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Deepfake Voice Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Audio Authenticity Verification '
    'Using Deep Learning and Prolog'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown(
    """
    This system analyses uploaded audio recordings using a
    Convolutional Neural Network (CNN) and a Prolog-based
    knowledge system.

    The CNN identifies patterns associated with genuine or
    synthetic speech, while Prolog provides rule-based
    reasoning and recommendations.
    """
)

st.divider()


# ============================================================
# AUDIO UPLOAD
# ============================================================

st.markdown(
    '<div class="section-heading">1. Upload Audio Recording</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Select an audio recording",
    type=["wav", "flac"],
    help="Supported formats: WAV and FLAC"
)


# ============================================================
# AUDIO ANALYSIS
# ============================================================

if uploaded_file is not None:

    file_size = len(uploaded_file.getvalue()) / 1024

    col1, col2 = st.columns(2)

    with col1:
        st.success("Audio uploaded successfully")

    with col2:
        st.caption(
            f"File size: {file_size:.2f} KB"
        )

    st.write(f"**File name:** {uploaded_file.name}")

    st.audio(uploaded_file)

    analyse_button = st.button(
        "Analyse Audio",
        type="primary",
        use_container_width=True
    )

    if analyse_button:

        temporary_path = None

        try:

            suffix = Path(uploaded_file.name).suffix.lower()

            # Save uploaded audio temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temporary_file:

                temporary_file.write(
                    uploaded_file.getvalue()
                )

                temporary_path = temporary_file.name

            # CNN and Prolog analysis
            with st.spinner(
                "Processing audio and performing intelligent analysis..."
            ):

                result = analyze_audio_with_prolog(
                    temporary_path
                )

            st.success("Audio analysis completed successfully!")

            st.divider()


            # ==================================================
            # CNN PREDICTION RESULTS
            # ==================================================

            st.markdown(
                '<div class="section-heading">'
                '2. CNN Prediction Results'
                '</div>',
                unsafe_allow_html=True
            )

            prediction = result["cnn_prediction"]

            if prediction == "spoof":

                st.markdown(
                    """
                    <div class="result-card spoof-card">
                        <div class="result-label">
                            SPOOF AUDIO DETECTED
                        </div>
                        <p>Patterns associated with synthetic
                        or manipulated speech were detected.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-card genuine-card">
                        <div class="result-label">
                            BONA FIDE AUDIO
                        </div>
                        <p>The audio appears consistent with
                        genuine human speech.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # Probability values
            bona_fide_probability = result[
                "bona_fide_probability"
            ]

            spoof_probability = result[
                "spoof_probability"
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Bona Fide Probability",
                    f"{bona_fide_probability * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, bona_fide_probability))
                )

            with col2:

                st.metric(
                    "Spoof Probability",
                    f"{spoof_probability * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, spoof_probability))
                )

            st.caption(
                f"Decision threshold: {result['threshold']:.4f}"
            )

            st.caption(
                "Probabilities are model scores and are not "
                "necessarily calibrated confidence estimates."
            )

            st.divider()


            # ==================================================
            # PROLOG REASONING
            # ==================================================

            st.markdown(
                '<div class="section-heading">'
                '3. Prolog Knowledge-Based Reasoning'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write("**Classification**")

                st.info(
                    str(
                        result["prolog_classification"]
                    ).replace("_", " ").upper()
                )

            with col2:

                st.write("**Confidence Level**")

                st.info(
                    str(result["confidence"]).upper()
                )

            st.write("**Audio Assessment**")

            assessment = (
                str(result["assessment"])
                .replace("_", " ")
                .title()
            )

            st.write(assessment)

            st.write("**System Recommendation**")

            st.info(result["recommendation"])

            st.divider()


            # ==================================================
            # FINAL ANALYSIS
            # ==================================================

            st.markdown(
                '<div class="section-heading">'
                '4. Final Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            if result["assessment"] == "suspicious":

                st.error(
                    "The audio contains patterns associated "
                    "with synthetic or manipulated speech. "
                    "Further verification is recommended."
                )

            elif result["assessment"] == "likely_genuine":

                st.success(
                    "The audio appears consistent with "
                    "genuine human speech. This is not a "
                    "guarantee of authenticity."
                )

            elif result["assessment"] == "requires_verification":

                st.warning(
                    "The audio appears potentially genuine, "
                    "but additional verification is recommended."
                )

            else:

                st.warning(
                    "The system cannot make a sufficiently "
                    "reliable assessment. Further verification "
                    "is required."
                )

            st.caption(
                "This system provides an automated assessment "
                "and does not constitute definitive forensic "
                "proof of audio authenticity."
            )


        except Exception as error:

            st.error(
                "Audio analysis failed. Please check the "
                "recording and try again."
            )

            with st.expander("Technical error details"):

                st.exception(error)

        finally:

            # Remove temporary audio file
            if temporary_path and Path(temporary_path).exists():

                Path(temporary_path).unlink()


else:

    st.info(
        "Upload a WAV or FLAC audio recording "
        "to begin the analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>Deepfake Voice Detection System</b><br>
        CNN | Audio Signal Processing | Prolog<br>
        Intelligent Audio Authenticity Assessment
    </div>
    """,
    unsafe_allow_html=True
)