import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Intelligent IDS",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Intelligent Intrusion Detection System")
st.write("Real-time network traffic analysis using Machine Learning")

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Info")
st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Dataset:** CICIDS2017")
st.sidebar.write("**Task:** Attack vs Benign")

# ---------------- Main ----------------
tab1, tab2 = st.tabs(["Manual Input", "JSON Input"])

default_features = [0.0] * 80

with tab1:
    st.subheader("Enter Network Features")

    col1, col2 = st.columns(2)

    with col1:
        f1 = st.number_input("Feature 1", value=0.0)
        f2 = st.number_input("Feature 2", value=0.0)
        f3 = st.number_input("Feature 3", value=0.0)
        f4 = st.number_input("Feature 4", value=0.0)
        f5 = st.number_input("Feature 5", value=0.0)
        f6 = st.number_input("Feature 6", value=0.0)
        f7 = st.number_input("Feature 7", value=0.0)
        f8 = st.number_input("Feature 8", value=0.0)

    with col2:
        f9 = st.number_input("Feature 9", value=0.0)
        f10 = st.number_input("Feature 10", value=0.0)
        f11 = st.number_input("Feature 11", value=0.0)
        f12 = st.number_input("Feature 12", value=0.0)
        f13 = st.number_input("Feature 13", value=0.0)
        f14 = st.number_input("Feature 14", value=0.0)
        f15 = st.number_input("Feature 15", value=0.0)
        f16 = st.number_input("Feature 16", value=0.0)

    manual_features = [
        f1, f2, f3, f4, f5, f6, f7, f8,
        f9, f10, f11, f12, f13, f14, f15, f16
    ]

    features_to_send = manual_features + [0.0] * (80 - len(manual_features))

    st.info("Only the first 16 features are entered manually here. The remaining features are filled with 0.0 for demo purposes.")

with tab2:
    st.subheader("Paste JSON Input")
    json_text = st.text_area(
        "JSON",
        value=json.dumps({"features": default_features}, indent=2),
        height=250
    )

    try:
        parsed_json = json.loads(json_text)
        features_to_send = parsed_json.get("features", default_features)

        if not isinstance(features_to_send, list):
            st.error("The 'features' value must be a list.")
            features_to_send = default_features
        else:
            st.success(f"Loaded {len(features_to_send)} features.")
    except json.JSONDecodeError:
        st.error("Invalid JSON format.")
        features_to_send = default_features

# ---------------- Prediction ----------------
st.markdown("---")

if st.button("Analyze Traffic"):
    if len(features_to_send) != 80:
        st.error(f"Your model expects 80 features, but you provided {len(features_to_send)}.")
    else:
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"features": features_to_send},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", 0.0)

                st.subheader("Prediction Result")

                if prediction == "Attack Detected":
                    st.error(f"🚨 {prediction}")
                else:
                    st.success(f"✅ {prediction}")

                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.progress(min(max(float(confidence), 0.0), 1.0))

                st.json(result)

            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"Unexpected error: {e}")