import gradio as gr
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model/placement_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict(
    iq,
    prev_sem,
    cgpa,
    academic,
    internship,
    extra_curricular,
    communication,
    projects
):
    # Create input dataframe
    data = pd.DataFrame({
        "IQ": [iq],
        "Prev_Sem_Result": [prev_sem],
        "CGPA": [cgpa],
        "Academic_Performance": [academic],
        "Internship_Experience": [internship],
        "Extra_Curricular_Score": [extra_curricular],
        "Communication_Skills": [communication],
        "Projects_Completed": [projects]
    })

    # One-Hot Encode
    data = pd.get_dummies(data)

    # IMPORTANT:
    # We will improve this in the next step by saving
    # the training columns so the app always matches
    # the model exactly.

    try:
        scaled = scaler.transform(data)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled).max() * 100

        if prediction == "Yes":
            result = "✅ Placement Likely"
        else:
            result = "❌ Placement Unlikely"

        return result, f"{probability:.2f}%"

    except Exception as e:
        return "Prediction Error", str(e)


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="IQ"),
        gr.Number(label="Previous Semester Result"),
        gr.Number(label="CGPA"),
        gr.Number(label="Academic Performance"),
        gr.Dropdown(
            ["Yes", "No"],
            label="Internship Experience"
        ),
        gr.Number(label="Extra Curricular Score"),
        gr.Number(label="Communication Skills"),
        gr.Number(label="Projects Completed"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
    ],
    title="🎓 Student Placement Prediction",
    description="Predict whether a student is likely to be placed using a Machine Learning model.",
)

demo.launch()