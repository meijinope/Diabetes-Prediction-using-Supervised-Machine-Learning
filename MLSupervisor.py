import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# local imports
from utilities.dataset import load_dataset, append_to_csv, REQUIRED_COLUMNS
from utilities.helpers import validate_and_cast
from models.knn import build_knn
from models.svm import build_svm
from models.ann import build_ann

app = Flask(__name__)

label_encoder = None
models = {}
class_names = []
feature_columns = REQUIRED_COLUMNS[:-1]

def train_models():
    global label_encoder, models, class_names

    df = load_dataset().copy()
    X = df[feature_columns].values
    y_raw = df["Diagnosis"].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build and fit each model
    knn = build_knn().fit(X_train, y_train)
    svm = build_svm().fit(X_train, y_train)
    ann = build_ann().fit(X_train, y_train)

    # Log accuracy
    for name, model in [("KNN", knn), ("SVM", svm), ("ANN", ann)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name}] Accuracy: {acc:.3f}")
        try:
            print(classification_report(y_test, y_pred, target_names=class_names))
        except:
            pass

    models.update({"KNN": knn, "SVM": svm, "ANN": ann})

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, probs, error = None, None, None
    model_results = {}
    model_probs = {}

    if request.method == "POST":
        try:
            values = validate_and_cast(request.form)
            X_infer = np.array(values, dtype=float).reshape(1, -1)

            # --- 1) Run all models ---
            for name, model in models.items():
                y_hat = model.predict(X_infer)[0]
                pred_label = label_encoder.inverse_transform([y_hat])[0]
                model_results[name] = pred_label

                if hasattr(model.named_steps["clf"], "predict_proba"):
                    proba = model.predict_proba(X_infer)[0]
                    model_probs[name] = [
                        (class_names[i], float(proba[i])) for i in range(len(class_names))
                    ]

            # --- 2) Final result (majority voting) ---
            preds = list(model_results.values())
            prediction = max(set(preds), key=preds.count)

            # --- 3) Insert only final result into CSV ---
            row = dict(zip(feature_columns, values))
            row["Diagnosis"] = prediction
            append_to_csv(row)

        except Exception as e:
            error = str(e)

    return render_template(
        "main.html",
        prediction=prediction,
        model_results=model_results,
        model_probs=model_probs,
        model_names=list(models.keys()),
        class_names=class_names,
        error=error
    )

if __name__ == "__main__":
    train_models()
    app.run(debug=True)
