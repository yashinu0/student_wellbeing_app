from flask import (
    Flask,
    render_template,
    request,
    session,
    flash,
    jsonify,
    redirect,
    url_for,
)
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import joblib
import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from datetime import datetime
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG FLASK
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "change-me"  # ⚠️ à changer en prod (clé secrète forte)

# -----------------------------------------------------------------------------
# CHEMINS MODELES & FICHIERS
# -----------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "clf_depression.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_cluster.pkl")
SEVERITY_PATH = os.path.join(MODEL_DIR, "severity_regressor.pkl")
CLUSTER_RECO_PATH = os.path.join(MODEL_DIR, "cluster_recommendations.json")
CLUSTER_AUTO_RECO_PATH = os.path.join(MODEL_DIR, "cluster_auto_reco.json")
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
FEATURE_RECO_PATH = os.path.join(MODEL_DIR, "feature_recomendation.json")

DATA_DIR = "data"
SUBMISSION_CSV = os.path.join(DATA_DIR, "submissions.csv")
ANOMALY_RESULTS_CSV = os.path.join(DATA_DIR, "anomaly_results.csv")

USERS_FILE = "users.json"

# -----------------------------------------------------------------------------
# VARIABLES GLOBALES
# -----------------------------------------------------------------------------
load_error = None
kmeans_error = None
severity_error = None
anomaly_error = None
cluster_reco = {}
cluster_reco_auto = {}
feature_reco_rules = {"default": None, "rules": [], "optional_rules": []}

CLUSTER_LABELS = {
    0: "High Academic Pressure, High Risk",
    1: "Balanced & Satisfied, Low Risk",
    2: "High Pressure on Student",
    3: "Very Satisfied, Low Sleep (Low Risk)",
    4: "Moderately Satisfied, Medium Stress (Low Risk)",
}

FEATURES = [
    "Gender",
    "Age",
    "Profession",
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "cluster",
]

CLUSTER_FEATURES = [
    "Academic Pressure",
    "Sleep Duration",
    "Financial Stress",
    "Study Satisfaction",
]

HIGH_STRESS_CLUSTERS = {0, 2}
STABLE_CLUSTERS = {1, 4}

AGG_FEATURE_COLUMNS = [
    "avg_academic_pressure",
    "avg_sleep_duration",
    "avg_financial_stress",
    "avg_study_satisfaction",
    "depression_rate",
    "high_risk_ratio",
    "avg_severity_score",
    "cluster_high_stress_ratio",
    "cluster_stable_ratio",
]

# -----------------------------------------------------------------------------
# GESTION UTILISATEURS (JSON)
# -----------------------------------------------------------------------------
def load_users():
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Your users.json format: {"users": [ ... ]}
            if isinstance(data, dict) and "users" in data and isinstance(data["users"], list):
                return data["users"]

            # If someday you store it directly as a list
            if isinstance(data, list):
                return data

            return []
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def compute_age_from_dob(dob_str):
    """Very simple age computation from 'YYYY-MM-DD'."""
    try:
        birth = datetime.strptime(dob_str, "%Y-%m-%d").date()
        today = datetime.today().date()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except Exception:
        return None

def save_users(users):
    """Sauvegarde la liste complète des utilisateurs dans users.json."""
    data = {"users": users}
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def find_user_by_email(email):
    """Retourne l'utilisateur correspondant à l'email, ou None."""
    users = load_users()  # now guaranteed to be a LIST
    for u in users:
        if isinstance(u, dict) and u.get("email") == email:
            return u
    return None

def get_current_user():
    email = session.get("user_email")
    if not email:
        return None
    return find_user_by_email(email)


# -----------------------------------------------------------------------------
# CHARGEMENT DES MODELES ML
# -----------------------------------------------------------------------------
try:
    clf = joblib.load(MODEL_PATH)
except Exception as exc:
    clf = None
    load_error = f"Failed to load classifier: {exc}"

try:
    kmeans = joblib.load(KMEANS_PATH)
except Exception as exc:
    kmeans = None
    kmeans_error = f"Failed to load KMeans: {exc}"

try:
    severity_model = joblib.load(SEVERITY_PATH)
except Exception as exc:
    severity_model = None
    severity_error = f"Failed to load severity model: {exc}"

try:
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
except Exception as exc:
    anomaly_model = None
    anomaly_error = f"Failed to load anomaly detector: {exc}"

try:
    if os.path.exists(CLUSTER_RECO_PATH):
        with open(CLUSTER_RECO_PATH, "r", encoding="utf-8") as f:
            cluster_reco = json.load(f)
    if os.path.exists(CLUSTER_AUTO_RECO_PATH):
        with open(CLUSTER_AUTO_RECO_PATH, "r", encoding="utf-8") as f:
            cluster_reco_auto = json.load(f)
    if os.path.exists(FEATURE_RECO_PATH):
        with open(FEATURE_RECO_PATH, "r", encoding="utf-8") as f:
            feature_reco_rules = json.load(f)
except Exception:
    # do not hard-fail if JSON malformed
    cluster_reco = {}
    cluster_reco_auto = {}
    feature_reco_rules = {"default": None, "rules": [], "optional_rules": []}


def generate_feature_recommendations(data: dict, model_label: str):
    """
    Apply feature_recomendation.json rules to incoming data and return a list of recommendations.
    """

    def _feature_value(name: str):
        mapping = {
            "Academic Pressure": data.get("academic_pressure"),
            "Sleep Duration": data.get("sleep_duration"),
            "Financial Stress": data.get("financial_stress"),
            "Study Satisfaction": data.get("study_satisfaction"),
            "Work/Study Hours": data.get("work_hours"),
            "Depression": 1 if str(model_label).lower().startswith("at risk") else 0,
        }
        val = mapping.get(name)
        # Align sleep encoding if client sends 1..4 instead of 0..3
        if name == "Sleep Duration" and val is not None:
            try:
                val = float(val) - 1
            except Exception:
                pass
        return val

    def _matches_rule(value, op, target):
        try:
            if value is None:
                return False
            if op == ">":
                return value > target
            if op == ">=":
                return value >= target
            if op == "<":
                return value < target
            if op == "<=":
                return value <= target
            if op == "==":
                return value == target
        except Exception:
            return False
        return False

    collected = []
    for rule in feature_reco_rules.get("rules", []):
        feat = rule.get("feature")
        op = rule.get("operator")
        target = rule.get("value")
        if not feat or not op:
            continue
        value = _feature_value(feat)
        if _matches_rule(value, op, target):
            collected.append((rule.get("priority", 0), rule.get("recommendation")))

    for rule in feature_reco_rules.get("optional_rules", []):
        feat = rule.get("feature")
        op = rule.get("operator")
        target = rule.get("value")
        if not feat or not op:
            continue
        value = _feature_value(feat)
        if _matches_rule(value, op, target):
            collected.append((rule.get("priority", 0), rule.get("recommendation")))

    collected.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    recs = []
    for _, rec in collected:
        if rec and rec not in seen:
            seen.add(rec)
            recs.append(rec)

    if not recs and feature_reco_rules.get("default"):
        recs.append(feature_reco_rules["default"])

    return recs

# -----------------------------------------------------------------------------
# FONCTIONS UTILITAIRES ML
# -----------------------------------------------------------------------------
def build_feature_vector(form):
    def get_float(name, default=0.0):
        val = form.get(name, "")
        try:
            return float(val)
        except Exception:
            return default

    values = [
        get_float("gender"),
        get_float("age"),
        get_float("profession"),
        get_float("academic_pressure"),
        get_float("cgpa"),
        get_float("study_satisfaction"),
        get_float("sleep_duration"),
        get_float("dietary_habits"),
        get_float("degree"),
        get_float("work_hours"),
        get_float("financial_stress"),
        get_float("family_history"),
    ]
    values.append(0.0)  # cluster placeholder
    return np.array([values])


def build_cluster_vector(form):
    """Use only the 4 cluster features, in the right order."""
    def get_float(name, default=0.0):
        val = form.get(name, "")
        try:
            return float(val)
        except Exception:
            return default

    return np.array(
        [
            [
                get_float("academic_pressure"),
                get_float("sleep_duration"),
                get_float("financial_stress"),
                get_float("study_satisfaction"),
            ]
        ]
    )


def make_recommendations(y_proba: float):
    if y_proba < 0.3:
        return "Low"
    elif y_proba < 0.7:
        return "Medium"
    return "High"


def full_assessment(form):
    """
    Run classification, severity regression, clustering, and map recommendations.
    form = request.form (HTML) ou dict JSON (API).
    """
    x = build_feature_vector(form)
    x_cluster = build_cluster_vector(form)

    # Clustering
    if kmeans is not None:
        cluster_id = int(kmeans.predict(x_cluster)[0])
        x[:, -1] = cluster_id  # inject cluster for downstream models
    else:
        cluster_id = 0

    # Classification
    y_pred = clf.predict(x)[0]
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(x)[0, 1]
    else:
        y_proba = float(y_pred)
    y_proba = float(np.clip(y_proba, 0.0, 1.0))

    # Severity
    severity_score = None
    if severity_model is not None:
        try:
            severity_score = float(severity_model.predict(x)[0])
        except Exception:
            severity_score = None

    risk_level = make_recommendations(y_proba)

    cid_key = str(cluster_id)
    reco_fixed = cluster_reco.get(cid_key, cluster_reco.get(cluster_id))
    cluster_name = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")

    return {
        "label": "At risk of depression" if y_pred == 1 else "Not at risk of depression",
        "proba": float(y_proba),
        "risk_level": risk_level,
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "reco_fixed": reco_fixed,
        "severity": severity_score,
    }

# -----------------------------------------------------------------------------
# DATA LOGGING & ANOMALY UTILS
# -----------------------------------------------------------------------------
SUBMISSION_FIELDS = [
    "gender",
    "age",
    "profession",
    "academic_pressure",
    "cgpa",
    "study_satisfaction",
    "sleep_duration",
    "dietary_habits",
    "degree",
    "work_hours",
    "financial_stress",
    "family_history",
]


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def sanitize_submission_features(data: dict) -> dict:
    """Extracts and converts expected submission fields to float where possible."""
    features = {}
    for field in SUBMISSION_FIELDS:
        features[field] = _to_float(data.get(field))
    return features


def append_submission_record(features: dict, result: dict):
    """Append one submission row to CSV with model outputs and timestamp."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        now = datetime.now()
        base_row = sanitize_submission_features(features)
        row = {
            **base_row,
            "predicted_risk": result.get("label"),
            "predicted_probability": result.get("proba"),
            "severity_score": result.get("severity"),
            "cluster_id": result.get("cluster_id"),
            "risk_level": result.get("risk_level"),
            "risk_flag": 1 if str(result.get("label", "")).lower().startswith("at risk") else 0,
            "high_risk_flag": 1 if str(result.get("risk_level", "")).lower() == "high" else 0,
            "submission_year": now.year,
            "submission_month": now.month,
        }
        df_new = pd.DataFrame([row])
        header = not os.path.exists(SUBMISSION_CSV)
        df_new.to_csv(SUBMISSION_CSV, mode="a", header=header, index=False)
    except Exception:
        # Failing to log should not break the user flow
        pass


def load_submissions_df():
    if not os.path.exists(SUBMISSION_CSV):
        return None
    try:
        return pd.read_csv(SUBMISSION_CSV)
    except Exception:
        return None


def compute_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate submissions by (year, month) and compute required metrics."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["risk_flag"] = df.get("risk_flag", 0).fillna(0)
    df["high_risk_flag"] = df.get("high_risk_flag", 0).fillna(0)
    df["severity_score"] = df.get("severity_score", 0).fillna(0)
    df["cluster_id"] = df.get("cluster_id", -1).fillna(-1)

    for col in ["academic_pressure", "sleep_duration", "financial_stress", "study_satisfaction"]:
        if col in df:
            df[col] = df[col].fillna(0)

    df["cluster_high_stress"] = df["cluster_id"].apply(lambda c: 1 if c in HIGH_STRESS_CLUSTERS else 0)
    df["cluster_stable"] = df["cluster_id"].apply(lambda c: 1 if c in STABLE_CLUSTERS else 0)

    grouped = df.groupby(["submission_year", "submission_month"], as_index=False).agg(
        avg_academic_pressure=("academic_pressure", "mean"),
        avg_sleep_duration=("sleep_duration", "mean"),
        avg_financial_stress=("financial_stress", "mean"),
        avg_study_satisfaction=("study_satisfaction", "mean"),
        depression_rate=("risk_flag", "mean"),
        high_risk_ratio=("high_risk_flag", "mean"),
        avg_severity_score=("severity_score", "mean"),
        cluster_high_stress_ratio=("cluster_high_stress", "mean"),
        cluster_stable_ratio=("cluster_stable", "mean"),
    )

    return grouped


def run_anomaly_detection():
    """Compute monthly aggregates and apply anomaly detector."""
    df = load_submissions_df()
    agg_df = compute_monthly_aggregates(df)
    if agg_df.empty:
        return {"success": False, "message": "No submissions available."}
    if anomaly_model is None:
        return {"success": False, "message": anomaly_error or "Anomaly model not loaded."}
    
     # Round aggregated numeric features to 2 decimals before persistence/model
    try:
        agg_df[AGG_FEATURE_COLUMNS] = agg_df[AGG_FEATURE_COLUMNS].round(2)
    except Exception:
        pass

    # Persist the aggregated dataset (pre-anomaly) for audit/troubleshooting
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        agg_df.to_csv(ANOMALY_RESULTS_CSV, index=False)
    except Exception:
        pass

    features_df = agg_df[AGG_FEATURE_COLUMNS].copy()
    try:
        scores = anomaly_model.decision_function(features_df)
        preds = anomaly_model.predict(features_df)
        threshold = float(pd.Series(scores).quantile(0.10))
        agg_df["anomaly_score"] = scores
        agg_df["anomaly_flag"] = (scores < threshold).astype(int)
        
        # Persist the enriched dataset with anomaly outputs
        try:
            agg_df.to_csv(ANOMALY_RESULTS_CSV, index=False)
        except Exception:
            pass
    except Exception as exc:
        return {"success": False, "message": f"Anomaly detection failed: {exc}"}

    return {
        "success": True,
        "data": agg_df.to_dict(orient="records"),
        "columns": ["submission_year", "submission_month"] + AGG_FEATURE_COLUMNS + ["anomaly_flag", "anomaly_score"],
    }


def _model_status(name, path, model_obj, load_err):
    """Return status info for a model artifact."""
    exists = os.path.exists(path)
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if exists else "N/A"
    loaded = model_obj is not None and load_err is None
    status = "OK" if loaded else ("Failed" if load_err else ("Missing" if not exists else "Warning"))
    message = "Loaded" if loaded else (load_err or ("File not found" if not exists else "Not loaded"))
    return {
        "name": name,
        "path": path,
        "status": status,
        "message": message,
        "updated_at": mtime,
    }

# -----------------------------------------------------------------------------
# ROUTES PAGES (VUES HTML)
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """
    Page d'accueil → index1.html
    """
    if load_error:
        flash(load_error)
    if kmeans_error:
        flash(kmeans_error)
    if severity_error:
        flash(severity_error)
    return render_template("index1.html", load_error=load_error)


@app.route("/register", methods=["GET", "POST"])
def register():
    """Registration for students, scientists, or admins."""
    if "user_email" in session:
        flash("Vous êtes déjà connecté. Déconnectez-vous pour créer un nouveau compte.")
        role = session.get("user_role")
        if role == "scientist":
            return redirect(url_for("datascientist_page"))
        if role == "admin":
            return redirect(url_for("admin_page"))
        return redirect(url_for("student_page"))

    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        dob = request.form.get("dob")
        role = request.form.get("role")  # scientist, student/other, or admin
        gender = request.form.get("gender")
        profession = request.form.get("profession")
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Email et mot de passe sont obligatoires.")
            return redirect(url_for("register"))

        if find_user_by_email(email) is not None:
            flash("Un compte avec cet email existe déjà.")
            return redirect(url_for("register"))

        users = load_users()
        users.append({
            "first_name": first_name,
            "last_name": last_name,
            "dob": dob,
            "role": role,
            "gender": gender,
            "profession": profession,
            "email": email,
            "password": generate_password_hash(password)
        })
        save_users(users)

        flash("Compte créé avec succès ! Vous pouvez maintenant vous connecter.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login and redirect to the appropriate role portal."""
    if "user_email" in session:
        flash("Vous êtes déjà connecté. Déconnectez-vous pour changer de compte.")
        role = session.get("user_role")
        if role == "scientist":
            return redirect(url_for("datascientist_page"))
        if role == "admin":
            return redirect(url_for("admin_page"))
        return redirect(url_for("student_page"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = find_user_by_email(email)

        if user and check_password_hash(user["password"], password):
            session["user_email"] = user["email"]
            session["user_role"] = user["role"]
            session["user_name"] = user.get("first_name", "")

            if user["role"] == "scientist":
                return redirect(url_for("datascientist_page"))
            if user["role"] == "admin":
                return redirect(url_for("admin_page"))
            return redirect(url_for("student_page"))

        flash("Identifiants incorrects.")
        return redirect(url_for("login"))

    return render_template("login.html")



@app.route("/logout")
def logout():
    """
    Déconnexion utilisateur.
    """
    session.clear()
    flash("Vous avez été déconnecté.")
    return redirect(url_for("index"))


@app.route("/student", methods=["GET"])
def student_page():
    user = get_current_user()
    if not user or user.get("role") not in ("student", "other"):
        # Only students can see this page
        return redirect(url_for("login"))

    # Prepare numeric features for the model from the user profile
    # You may already store them as numeric codes in users.json – if so, just use them directly
    gender = user.get("gender_num")  # e.g. 0 / 1, or adapt
    profession = user.get("profession_num")
    age = user.get("age_num")

    # If not stored, derive them:
    if age is None and user.get("dob"):
        age = compute_age_from_dob(user["dob"])

    # Example: convert string gender to numeric if needed
    if gender is None:
        g = (user.get("gender") or "").lower()
        if g in ("male", "m"):
            gender = 1
        elif g in ("female", "f"):
            gender = 0

    # Example: profession mapping if you stored it as a string
    if profession is None:
        prof = (user.get("profession") or "").lower()
        mapping = {
            "student": 0,
            "employee": 1,
            "unemployed": 2,
            "freelancer": 3,
        }
        profession = mapping.get(prof)

    # Pass these to the template so we can send them as hidden fields
    profile_features = {
        "gender": gender,
        "age": age,
        "profession": profession,
    }

    return render_template("student.html", profile=profile_features)


@app.route("/datascientist")
def datascientist_page():
    """
    Portail data scientist.
    """
    if "user_email" not in session or session.get("user_role") not in ("scientist", "admin"):
        return redirect(url_for("login"))
    return render_template("datascientist.html")


@app.route("/admin")
def admin_page():
    """
    Simple admin portal placeholder.
    """
    if session.get("user_role") != "admin":
        flash("Admin access required.")
        return redirect(url_for("login"))
    return render_template("admin.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Formulaire /predict classique (HTML) + résultat sur result.html.
    """
    if request.method == "GET":
        if load_error:
            flash(load_error)
        if kmeans_error:
            flash(kmeans_error)
        if severity_error:
            flash(severity_error)
        return render_template("predict.html", load_error=load_error)

    if clf is None:
        flash("Model is not available. Check server logs.")
        return render_template("predict.html"), 500

    # Calcul ML
    result = full_assessment(request.form)

    # Log submission to CSV
    form_features = {key: request.form.get(key) for key in SUBMISSION_FIELDS}
    append_submission_record(form_features, result)

    # Feature-based recommendations (server-side)
    feature_payload = sanitize_submission_features(form_features)
    custom_recs = generate_feature_recommendations(feature_payload, result["label"])

    # Récupérer les indicateurs individuels depuis le formulaire
    stress_academic = request.form.get("academic_pressure")
    sleep_quality = request.form.get("sleep_duration")
    wellbeing = request.form.get("study_satisfaction")

    # Convert severity to percentage for display consistency with /student
    severity_pct = None
    if result["severity"] is not None:
        severity_pct = float(result["severity"]) * 100

    return render_template(
        "result.html",
        label=result["label"],
        proba=f"{result['proba']:.3f}",
        risk_level=result["risk_level"],
        cluster_label=result["cluster_id"],
        cluster_name=result.get("cluster_name"),
        severity_score=(
            f"{severity_pct:.1f}%" if severity_pct is not None else None
        ),
        severity_pct=severity_pct,
        severity_raw=result["severity"],
        custom_recs=custom_recs,
        reco_fixed=result["reco_fixed"],
        reco_auto=None,
        # Pour la carte "Score détaillé"
        stress_academic=stress_academic,
        sleep_quality=sleep_quality,
        wellbeing=wellbeing,
    )

# -----------------------------------------------------------------------------
# ROUTES API (JSON) POUR student.html
# -----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API used by student.html.
    Expects all model features in the JSON body.
    """
    data = request.get_json() or {}

    # If some demographic features are missing, and user is logged in,
    # we can still try to fill them from the profile (defensive).
    user = get_current_user()
    if user:
        if "gender" not in data or data["gender"] in ("", None):
            data["gender"] = (
                user.get("gender_num")
                or (1 if (user.get("gender", "").lower() in ("male", "m")) else 0)
            )
        if "profession" not in data or data["profession"] in ("", None):
            data["profession"] = user.get("profession_num")
        if "age" not in data or data["age"] in ("", None):
            if user.get("age_num") is not None:
                data["age"] = user["age_num"]
            elif user.get("dob"):
                data["age"] = compute_age_from_dob(user["dob"])

    # Run the same pipeline as in /predict
    try:
        result = full_assessment(data)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    # Log submission to CSV
    append_submission_record(data, result)

    # Convert severity from 0..1 to 0..100 for the gauge in student.html
    severity_pct = round(float(result["severity"]) * 100, 1)

    return jsonify({
        "success": True,
        "risk_level": result["risk_level"],      # "Low / Moderate / High" etc.
        "severity_score": severity_pct,          # 0..100
        "cluster_label": result["cluster_name"], # e.g. "Resilient Balanced"
        # Optional extra fields if you want:
        "model_label": result["label"],
        "severity_raw": result["severity"],
    })



@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    x_cluster = build_cluster_vector(data)
    cluster_id = int(kmeans.predict(x_cluster)[0]) if kmeans is not None else 0

    return jsonify(
        {
            "cluster": cluster_id,
            "profile_name": CLUSTER_LABELS.get(
                cluster_id, f"Cluster {cluster_id}"
            ),
            "description": cluster_reco.get(str(cluster_id), "No description available."),
        }
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    recs = generate_feature_recommendations(data, data.get("model_label", ""))
    return jsonify({"recommendations": recs})

@app.route("/admin/anomaly", methods=["GET"])
def admin_anomaly():
    if session.get("user_role") != "admin":
        flash("Admin access required.")
        return redirect(url_for("login"))
    result = run_anomaly_detection()
    if not result.get("success"):
        return jsonify(result), 400
    return jsonify(result)


@app.route("/admin/export", methods=["GET"])
def admin_export():
    """Export the raw submissions CSV."""
    if session.get("user_role") != "admin":
        flash("Admin access required.")
        return redirect(url_for("login"))
    if not os.path.exists(SUBMISSION_CSV):
        return jsonify({"success": False, "message": "No submissions available."}), 404
    return send_file(SUBMISSION_CSV, as_attachment=True, download_name="submissions.csv")


@app.route("/admin/users", methods=["GET"])
def admin_users():
    """Return the user directory from users.json (sanitized)."""
    if session.get("user_role") != "admin":
        flash("Admin access required.")
        return redirect(url_for("login"))
    users = load_users()
    safe_users = []
    for u in users:
        if not isinstance(u, dict):
            continue
        safe_users.append({
            "first_name": u.get("first_name") or "",
            "last_name": u.get("last_name") or "",
            "email": u.get("email") or "",
            "role": u.get("role") or "",
            "dob": u.get("dob") or "",
            "profession": u.get("profession") or "",
            "gender": u.get("gender") or "",
        })
    return jsonify({"success": True, "users": safe_users})


@app.route("/admin/models", methods=["GET"])
def admin_models():
    """Return status for all ML artifacts."""
    if session.get("user_role") != "admin":
        flash("Admin access required.")
        return redirect(url_for("login"))
    statuses = [
        _model_status("classifier", MODEL_PATH, clf, load_error),
        _model_status("kmeans", KMEANS_PATH, kmeans, kmeans_error),
        _model_status("severity", SEVERITY_PATH, severity_model, severity_error),
        _model_status("anomaly", ANOMALY_MODEL_PATH, anomaly_model, anomaly_error),
    ]
    return jsonify({"success": True, "models": statuses})


# -----------------------------------------------------------------------------
# Metrics APIs for Data Scientist page
# -----------------------------------------------------------------------------
def _load_metrics_json(filename):
    path = os.path.join("artifacts", "metrics", filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.route("/api/metrics/classification", methods=["GET"])
def api_metrics_classification():
    data = _load_metrics_json("classification.json")
    if data is None:
        return jsonify({"success": False, "error": "classification metrics not found"}), 404
    return jsonify({"success": True, "data": data})


@app.route("/api/metrics/clustering", methods=["GET"])
def api_metrics_clustering():
    data = _load_metrics_json("clustering.json")
    if data is None:
        return jsonify({"success": False, "error": "clustering metrics not found"}), 404
    return jsonify({"success": True, "data": data})


@app.route("/api/metrics/regression", methods=["GET"])
def api_metrics_regression():
    data = _load_metrics_json("regression.json")
    if data is None:
        return jsonify({"success": False, "error": "regression metrics not found"}), 404
    return jsonify({"success": True, "data": data})


@app.route("/api/metrics/anomaly", methods=["GET"])
def api_metrics_anomaly():
    data = _load_metrics_json("anomaly.json")
    if data is None:
        return jsonify({"success": False, "error": "anomaly metrics not found"}), 404
    return jsonify({"success": True, "data": data})

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
