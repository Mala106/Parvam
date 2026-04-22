from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import csv
import os
import sqlite3
from datetime import date
from functools import wraps


app = Flask(__name__)
app.secret_key = "complaint_secret_2024"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "complaints.db")
CATEGORIES = ["Electricity", "Water", "Food", "Other"]
ADMIN_ACCOUNTS = [
    ("admin", "admin123", None),
    ("electricity_admin", "electric123", "Electricity"),
    ("water_admin", "water123", "Water"),
    ("food_admin", "food123", "Food"),
    ("other_admin", "other123", "Other"),
]


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=MEMORY")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                role     TEXT    NOT NULL DEFAULT 'user'
            );
            CREATE TABLE IF NOT EXISTS complaints (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                name        TEXT    NOT NULL,
                category    TEXT    NOT NULL,
                description TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'Pending',
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )

        user_columns = [row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "admin_category" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN admin_category TEXT")

        for username, password, admin_category in ADMIN_ACCOUNTS:
            conn.execute(
                "INSERT OR IGNORE INTO users(username,password,role,admin_category) VALUES(?,?,?,?)",
                (username, password, "admin", admin_category),
            )
            conn.execute(
                "UPDATE users SET password=?, role='admin', admin_category=? WHERE username=?",
                (password, admin_category, username),
            )

        conn.execute("INSERT OR IGNORE INTO users(username,password,role) VALUES('user','user123','user')")
        conn.execute("UPDATE users SET password='user123', role='user', admin_category=NULL WHERE username='user'")

        default_user = conn.execute("SELECT id FROM users WHERE username='user'").fetchone()
        user_id = default_user["id"]

        existing = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        if existing == 0:
            sample = [
                (user_id, "Alice", "Electricity", "Street light not working near Block-A", "2024-01-10", "Resolved"),
                (user_id, "Alice", "Water", "No water supply for 3 days", "2024-01-12", "Pending"),
                (user_id, "Bob", "Food", "Canteen food quality is very poor", "2024-01-14", "In Progress"),
                (user_id, "Alice", "Other", "Garbage not collected for a week", "2024-01-15", "Pending"),
                (user_id, "Bob", "Electricity", "Frequent power cuts in Hostel B", "2024-01-17", "Resolved"),
                (user_id, "Alice", "Water", "Water pipe burst on main road", "2024-01-18", "In Progress"),
                (user_id, "Bob", "Food", "Mess food contains insects", "2024-01-20", "Pending"),
                (user_id, "Alice", "Electricity", "No electricity in reading room", "2024-01-21", "Resolved"),
                (user_id, "Bob", "Other", "Wi-Fi not working in lab", "2024-01-22", "In Progress"),
                (user_id, "Alice", "Water", "Water tank overflowing causing damage", "2024-01-23", "Pending"),
                (user_id, "Bob", "Electricity", "Electricity bill discrepancy", "2024-01-25", "Resolved"),
                (user_id, "Alice", "Food", "Food served cold at dinner", "2024-01-26", "In Progress"),
                (user_id, "Bob", "Other", "Road potholes causing accidents", "2024-01-28", "Pending"),
                (user_id, "Alice", "Water", "Drinking water tastes bad", "2024-01-29", "Resolved"),
                (user_id, "Bob", "Electricity", "Short circuit in room 204", "2024-01-30", "In Progress"),
                (user_id, "Alice", "Food", "Portion size drastically reduced", "2024-02-01", "Pending"),
                (user_id, "Bob", "Other", "Loud noise from construction at night", "2024-02-03", "Resolved"),
                (user_id, "Alice", "Electricity", "AC not working in exam hall", "2024-02-05", "Pending"),
                (user_id, "Bob", "Water", "Water cooler broken in Block-C", "2024-02-07", "In Progress"),
                (user_id, "Alice", "Food", "No vegetarian options available", "2024-02-08", "Pending"),
                (user_id, "Bob", "Other", "Lift malfunction in main building", "2024-02-10", "Resolved"),
                (user_id, "Alice", "Water", "Sewage smell near dining hall", "2024-02-12", "Pending"),
                (user_id, "Bob", "Electricity", "Lights flickering in corridor", "2024-02-14", "In Progress"),
                (user_id, "Alice", "Food", "Overpriced items in canteen", "2024-02-15", "Pending"),
                (user_id, "Bob", "Other", "Parking area flooding in rain", "2024-02-17", "Resolved"),
                (user_id, "Alice", "Electricity", "Generator not starting during outage", "2024-02-19", "Pending"),
                (user_id, "Bob", "Water", "Low water pressure in showers", "2024-02-20", "In Progress"),
                (user_id, "Alice", "Food", "Unhygienic kitchen conditions observed", "2024-02-22", "Pending"),
                (user_id, "Bob", "Other", "CCTV cameras not functioning", "2024-02-24", "Resolved"),
                (user_id, "Alice", "Water", "Rainwater leaking into rooms", "2024-02-25", "Pending"),
            ]
            conn.executemany(
                "INSERT INTO complaints(user_id,name,category,description,date,status) VALUES(?,?,?,?,?,?)",
                sample,
            )


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)

    return decorated


@app.route("/", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    error = None
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            error = "Please enter both username and password."
        else:
            with get_db() as conn:
                user = conn.execute(
                    "SELECT * FROM users WHERE username=? AND password=?",
                    (username, password),
                ).fetchone()

            if user:
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["role"] = user["role"]
                return redirect(url_for("dashboard"))

            error = "Invalid username or password."

    return render_template("login.html", error=error, username=username)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    error = None
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not password or not confirm:
            error = "Please fill in all fields."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 4:
            error = "Password must be at least 4 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            try:
                with get_db() as conn:
                    cur = conn.execute(
                        "INSERT INTO users(username,password,role,admin_category) VALUES(?,?,?,NULL)",
                        (username, password, "user"),
                    )

                session["user_id"] = cur.lastrowid
                session["username"] = username
                session["role"] = "user"
                flash("Account created successfully.", "success")
                return redirect(url_for("dashboard"))
            except sqlite3.IntegrityError:
                error = "That username is already taken."

    return render_template("register.html", error=error, username=username)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        if session["role"] == "admin":
            admin = conn.execute("SELECT admin_category FROM users WHERE id=?", (session["user_id"],)).fetchone()
            admin_scope = admin["admin_category"] if admin else None
            cat = admin_scope or request.args.get("category", "")
            status = request.args.get("status", "")
            query = "SELECT * FROM complaints WHERE 1=1"
            params = []

            if cat:
                query += " AND category=?"
                params.append(cat)
            if status:
                query += " AND status=?"
                params.append(status)

            query += " ORDER BY id DESC"
            complaints = conn.execute(query, params).fetchall()

            stat_query = "SELECT COUNT(*) FROM complaints WHERE 1=1"
            stat_params = []
            if admin_scope:
                stat_query += " AND category=?"
                stat_params.append(admin_scope)
            total = conn.execute(stat_query, stat_params).fetchone()[0]
            pending = conn.execute(stat_query + " AND status='Pending'", stat_params).fetchone()[0]
            inprog = conn.execute(stat_query + " AND status='In Progress'", stat_params).fetchone()[0]
            resolved = conn.execute(stat_query + " AND status='Resolved'", stat_params).fetchone()[0]

            return render_template(
                "admin_dashboard.html",
                complaints=complaints,
                total=total,
                pending=pending,
                inprog=inprog,
                resolved=resolved,
                sel_cat=cat,
                sel_status=status,
                admin_scope=admin_scope,
                categories=CATEGORIES,
            )

        complaints = conn.execute(
            "SELECT * FROM complaints WHERE user_id=? ORDER BY id DESC",
            (session["user_id"],),
        ).fetchall()
        return render_template("user_dashboard.html", complaints=complaints)


@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit():
    if session["role"] == "admin":
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form["name"].strip()
        cat = request.form["category"]
        desc = request.form["description"].strip()
        incident_date = request.form["date"]

        if not all([name, cat, desc, incident_date]):
            flash("All fields are required.", "error")
        else:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO complaints(user_id,name,category,description,date,status) VALUES(?,?,?,?,?,?)",
                    (session["user_id"], name, cat, desc, incident_date, "Pending"),
                )
            flash("Complaint submitted successfully!", "success")
            return redirect(url_for("dashboard"))

    today = date.today().isoformat()
    return render_template("submit.html", today=today)


@app.route("/update_status/<int:cid>", methods=["POST"])
@login_required
@admin_required
def update_status(cid):
    new_status = request.form["status"]
    with get_db() as conn:
        admin = conn.execute("SELECT admin_category FROM users WHERE id=?", (session["user_id"],)).fetchone()
        admin_scope = admin["admin_category"] if admin else None
        if admin_scope:
            complaint = conn.execute("SELECT category FROM complaints WHERE id=?", (cid,)).fetchone()
            if not complaint or complaint["category"] != admin_scope:
                flash("This complaint is assigned to another admin.", "error")
                return redirect(url_for("dashboard"))
        conn.execute("UPDATE complaints SET status=? WHERE id=?", (new_status, cid))

    flash("Status updated.", "success")
    return redirect(
        url_for(
            "dashboard",
            category=request.args.get("category", ""),
            status=request.args.get("status", ""),
        )
    )


@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    """Return an ML category suggestion when trained model files exist."""
    try:
        import joblib
        from ml.train_model import preprocess_text

        model = joblib.load(os.path.join(BASE_DIR, "ml", "category_model.pkl"))
        tfidf = joblib.load(os.path.join(BASE_DIR, "ml", "tfidf_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(BASE_DIR, "ml", "label_encoder.pkl"))

        text = (request.get_json(silent=True) or {}).get("text", "")
        vec = tfidf.transform([preprocess_text(text)])
        pred = model.predict(vec)[0]
        category = label_encoder.inverse_transform([pred])[0]
        return jsonify({"category": category})
    except Exception:
        return jsonify({"category": None})


@app.route("/export_csv")
@login_required
@admin_required
def export_csv():
    with get_db() as conn:
        admin = conn.execute("SELECT admin_category FROM users WHERE id=?", (session["user_id"],)).fetchone()
        admin_scope = admin["admin_category"] if admin else None
        if admin_scope:
            rows = conn.execute(
                "SELECT id,user_id,name,category,description,date,status FROM complaints WHERE category=?",
                (admin_scope,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id,user_id,name,category,description,date,status FROM complaints"
            ).fetchall()

    path = os.path.join(BASE_DIR, "ml", "complaints_export.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_id", "name", "category", "description", "date", "status"])
        writer.writerows(rows)

    flash(f"Exported {len(rows)} records to ml/complaints_export.csv", "success")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
