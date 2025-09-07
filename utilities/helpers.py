def validate_and_cast(form):
    try:
        values = [
            int(form["Pregnancies"]),
            float(form["Glucose"]),
            float(form["BloodPressure"]),
            float(form["SkinThickness"]),
            float(form["Insulin"]),
            float(form["BMI"]),
            float(form["DiabetesPedigreeFunction"]),
            int(form["Age"]),
        ]
    except Exception:
        raise ValueError("Please ensure all fields are filled with valid numbers.")

    if values[1] <= 0 or values[5] <= 0 or values[7] <= 0:
        raise ValueError("Glucose, BMI, and Age must be positive.")
    return values
