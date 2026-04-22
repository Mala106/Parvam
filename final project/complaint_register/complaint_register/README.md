# 🗂️ Complaint Register

A Flask-based Complaint Management System with SQLite database,
role-based access control, and ML-powered category auto-suggestion.

---

## 📁 Project Structure

```
complaint_register/
├── app.py                  ← Flask app (all routes)
├── requirements.txt
├── complaints.db           ← Auto-created on first run
├── templates/
│   ├── base.html           ← Shared nav + layout
│   ├── login.html
│   ├── admin_dashboard.html
│   ├── user_dashboard.html
│   └── submit.html         ← With ML auto-suggest JS
└── ml/
    ├── train_model.py      ← Full ML pipeline (Steps 2–4)
    ├── category_model.pkl  ← Created after training
    ├── tfidf_vectorizer.pkl
    └── label_encoder.pkl
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the web app
```bash
python app.py
```
Open → http://localhost:5000

Default accounts:
| Username | Password | Role  |
|----------|----------|-------|
| admin | admin123 | Super admin |
| electricity_admin | electric123 | Electricity admin |
| water_admin | water123 | Water admin |
| food_admin | food123 | Food admin |
| other_admin | other123 | Other admin |
| user | user123 | Demo user |

New users can also create their own account from the login page.

---

## 🤖 ML Pipeline (Steps 2–5)

### Step 2 — Export complaints to CSV
Log in as **admin** → click **"⬇ Export CSV"** button.  
This saves `ml/complaints_export.csv`.

### Step 3–4 — Train the model & save it
```bash
cd complaint_register
python ml/train_model.py
```

Output:
- Tests 3 algorithms (Logistic Regression, Naive Bayes, Random Forest)
- Picks the best one automatically
- Prints accuracy + classification report
- Saves model to `ml/category_model.pkl`

### Step 5 — Model is live in Flask
The `/api/predict` route is already wired up in `app.py`.  
Once `category_model.pkl` exists, the **Submit Complaint** form will
auto-suggest the category as the user types their description.

---

## 🔑 Role-Based Access

| Feature                    | Admin | User |
|----------------------------|-------|------|
| View all complaints        | ✅    | ❌   |
| Filter by category/status  | ✅    | ❌   |
| Update complaint status    | ✅    | ❌   |
| Export CSV                 | ✅    | ❌   |
| Submit complaints          | ❌    | ✅   |
| View own complaints        | ❌    | ✅   |
| ML category auto-suggest   | N/A   | ✅   |

---

## 🗄️ Database Schema

```sql
CREATE TABLE users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT    UNIQUE NOT NULL,
    password TEXT    NOT NULL,
    role     TEXT    NOT NULL DEFAULT 'user'
);

CREATE TABLE complaints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    category    TEXT    NOT NULL,   -- Electricity / Water / Food / Other
    description TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'Pending',
    FOREIGN KEY(user_id) REFERENCES users(id)
);
```

---

## 📊 ML Algorithm Comparison

| Algorithm          | Speed | Accuracy | Best For           |
|--------------------|-------|----------|--------------------|
| Logistic Regression| Fast  | ~85–90%  | Baseline + prod    |
| Naive Bayes        | Fastest| ~80–85% | Small datasets     |
| Random Forest      | Slow  | ~88–92%  | Imbalanced classes |

The pipeline trains all three and **auto-selects** the best one.
