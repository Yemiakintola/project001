
import streamlit as st
import pandas as pd
import joblib

# ✅ Import the feature engineering function
from feature_engineering_module import create_input_engineered_features_df

# ✅ Load the trained pipeline (which includes preprocessing and model)
model = joblib.load("best_student_performance_prediction_pipeline.pkl")

# ✅ Streamlit UI
st.title("🎓 Student GPA Predictor")
st.markdown("""Upload a CSV file with student data including features:( StudentID(from: 1001 to 3392;	Age:15 to 18; 	Gender: Female = 1, Male = 0;	Ethnicity: For Fulani=0, Hausa=1, Ibo=2, Yoruba=3,	ParentalEducation: none=0, Primary=1, secondary=2, first degree=3, higher degrees:4;	StudyTimeWeekly: in float;	Absences=0 to 30.000	Tutoring: 1=yes, 0=no	ParentalSupport:1 = yes,no = 0;	Extracurricular:1 = yes,no = 0	Sports:1 = yes,no = 0	Music: 1 = yes,no = 0;	Volunteering: 1 = yes,no = 0	GradeClass:
) to predict GPA.""")

# ✅ File upload
uploaded_file = st.file_uploader("Upload student data file (.csv)", type=["csv"])

if uploaded_file:
    # ✅ Read uploaded CSV
    input_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Uploaded Data")
    st.write(input_df.head())

    try:
        # ✅ Feature engineering only
        engineered_df = create_input_engineered_features_df(input_df)

        # ✅ Predict using the trained pipeline (it handles encoding, scaling, etc.)
        predictions = model.predict(engineered_df)

        # ✅ Show predictions
        input_df["Predicted GPA"] = predictions
        st.subheader("📊 Prediction Results")
        st.write(input_df)

        # ✅ Download button
        csv = input_df.to_csv(index=False)
        st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
