"""
STEP 2 -> Export CSV  (already done via /export_csv route)
STEP 3 -> ML Pipeline (run this script offline)
STEP 4 -> Save model with joblib

Usage:
    cd complaint_register
    python ml/train_model.py
"""

import os, re, csv, sqlite3
import pandas as pd
import joblib

# --- sklearn ----------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Built-in stopwords - no NLTK download required
STOP = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were','be',
    'been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by',
    'for','with','about','against','between','into','through','during','before',
    'after','above','below','to','from','up','down','in','out','on','off','over',
    'under','again','further','then','once','here','there','when','where','why',
    'how','all','both','each','few','more','most','other','some','such','no',
    'nor','not','only','own','same','so','than','too','very','s','t','can','will',
    'just','don','should','now','d','ll','m','o','re','ve','y','ain','ma',
}
DB     = os.path.join(os.path.dirname(__file__), '..', 'complaints.db')
MODEL_DIR = os.path.dirname(__file__)

# ==========================================================================
# STAGE 1 - LOAD DATA
# ==========================================================================
def load_data() -> pd.DataFrame:
    """Connect to SQLite, pull complaints table into a DataFrame."""
    print("\n[1/6] LOADING DATA from SQLite...")
    conn = sqlite3.connect(DB)
    df   = pd.read_sql_query("SELECT * FROM complaints", conn)
    conn.close()
    print(f"      Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df

# ==========================================================================
# STAGE 2 - CLEAN DATA
# ==========================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, normalize categories, fix types."""
    print("\n[2/6] CLEANING DATA...")
    before = len(df)
    df = df.dropna(subset=['description', 'category'])
    df = df.drop_duplicates()
    df['description'] = df['description'].str.strip()
    df['category']    = df['category'].str.strip().str.capitalize()
    df['date']        = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['category'].isin(['Electricity','Water','Food','Other'])]
    print(f"      Rows after cleaning: {len(df)} (dropped {before - len(df)})")
    print(f"      Category distribution:\n{df['category'].value_counts().to_string()}")
    return df.reset_index(drop=True)

# ==========================================================================
# STAGE 3 - PREPROCESS TEXT
# ==========================================================================
def preprocess_text(text: str) -> str:
    """Lowercase -> remove punctuation -> tokenize -> remove stopwords -> lemmatize."""
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP and len(t) > 2]
    return ' '.join(tokens)

def vectorize_text(df: pd.DataFrame):
    """Apply TF-IDF vectorization to the cleaned description column."""
    print("\n[3/6] PREPROCESSING & VECTORIZING...")
    df['clean_desc'] = df['description'].apply(preprocess_text)
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X     = tfidf.fit_transform(df['clean_desc'])
    print(f"      TF-IDF matrix shape: {X.shape}")
    return X, tfidf

def encode_labels(df: pd.DataFrame, column: str = 'category'):
    """Label-encode the target column."""
    le = LabelEncoder()
    y  = le.fit_transform(df[column])
    print(f"      Labels: {list(le.classes_)} -> {list(range(len(le.classes_)))}")
    return y, le

def split_data(X, y):
    """80/20 train-test split."""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================================================
# STAGE 4 - TRAIN MODEL
# ==========================================================================
def train_model(X_train, y_train, algorithm: str = 'logistic'):
    """Fit the chosen classifier."""
    print(f"\n[4/6] TRAINING MODEL  (algorithm={algorithm})...")
    models = {
        'logistic': LogisticRegression(max_iter=500, C=1.0),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'naive_bayes': MultinomialNB(alpha=0.5),
    }
    model = models.get(algorithm, models['logistic'])
    model.fit(X_train, y_train)
    print(f"      Trained: {model.__class__.__name__}")
    return model

# ==========================================================================
# STAGE 5 - EVALUATE
# ==========================================================================
def evaluate_model(model, X_test, y_test, le):
    """Print accuracy, classification report, and confusion matrix."""
    print("\n[5/6] EVALUATING MODEL...")
    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"\n      OK Accuracy: {accuracy*100:.1f}%\n")
    print("      Classification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))
    print("      Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, preds)
    header = "      " + "  ".join(f"{c[:4]:>6}" for c in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        lbl  = le.classes_[i][:4].ljust(6)
        vals = "  ".join(f"{v:6}" for v in row)
        print(f"      {lbl}  {vals}")

# ==========================================================================
# STAGE 6 - SAVE MODEL
# ==========================================================================
def save_model(model, tfidf, le):
    """Persist model + vectorizer + label encoder using joblib."""
    print("\n[6/6] SAVING MODEL...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'category_model.pkl'))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(le,    os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print(f"      Saved -> ml/category_model.pkl")
    print(f"      Saved -> ml/tfidf_vectorizer.pkl")
    print(f"      Saved -> ml/label_encoder.pkl")

# ==========================================================================
# PREDICTION HELPER (used by Flask /api/predict)
# ==========================================================================
def predict_category(text: str) -> str:
    """Load saved model and return predicted category for a complaint description."""
    model = joblib.load(os.path.join(MODEL_DIR, 'category_model.pkl'))
    tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    le    = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    clean = preprocess_text(text)
    vec   = tfidf.transform([clean])
    pred  = model.predict(vec)[0]
    return le.inverse_transform([pred])[0]

# ==========================================================================
# MAIN PIPELINE
# ==========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  COMPLAINT REGISTER - ML TRAINING PIPELINE")
    print("=" * 60)

    df           = load_data()
    df           = clean_data(df)
    X, tfidf     = vectorize_text(df)
    y, le        = encode_labels(df, 'category')
    X_tr,X_te,y_tr,y_te = split_data(X, y)

    # -- Train all 3 and pick best --------------------------------------
    best_model, best_acc = None, 0
    for algo in ['logistic', 'naive_bayes', 'random_forest']:
        m   = train_model(X_tr, y_tr, algo)
        acc = accuracy_score(y_te, m.predict(X_te))
        print(f"      -> {algo}: {acc*100:.1f}% accuracy")
        if acc > best_acc:
            best_acc, best_model = acc, m

    print(f"\n      Best model: {best_model.__class__.__name__} ({best_acc*100:.1f}%)")
    evaluate_model(best_model, X_te, y_te, le)
    save_model(best_model, tfidf, le)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE - Flask /api/predict is now active")
    print("=" * 60)

