import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import threading
import pickle
import os

class EmotionTestUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Classification Model - Test Suite")
        self.root.geometry("1000x900")
        self.root.configure(bg='#f0f0f0')
        
        # Model variables
        self.model = None
        self.vectorizer = None
        self.test_data = None
        self.emotions = None
        self.predictions = None
        
        self.create_widgets()
        self.auto_train_model()
        
    def create_widgets(self):
        """Create UI elements"""
        
        # Title
        title_label = ttk.Label(self.root, text="Emotion Classification - Test Suite", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== TOP SECTION: SINGLE TEST =====
        top_frame = ttk.LabelFrame(main_frame, text="Single Text Test", padding=10)
        top_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        ttk.Label(top_frame, text="Enter text to predict emotion:").pack(anchor=tk.W)
        self.input_text = tk.Text(top_frame, height=3, width=100)
        self.input_text.pack(fill=tk.X, pady=5)
        
        predict_btn = ttk.Button(top_frame, text="Predict Emotion", command=self.predict_single)
        predict_btn.pack(pady=5)
        
        # Result
        ttk.Label(top_frame, text="Result:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.single_result_text = ttk.Label(top_frame, text="", foreground="green", font=("Arial", 11))
        self.single_result_text.pack(anchor=tk.W, pady=5)
        
        # ===== MIDDLE SECTION: BATCH TEST =====
        middle_frame = ttk.LabelFrame(main_frame, text="Batch Test (Load Test Data)", padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        
        button_frame = ttk.Frame(middle_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        load_test_btn = ttk.Button(button_frame, text="Load Test File", command=self.load_test_file)
        load_test_btn.pack(side=tk.LEFT, padx=5)
        
        use_train_btn = ttk.Button(button_frame, text="Use Training Data (20%)", command=self.use_train_test_split)
        use_train_btn.pack(side=tk.LEFT, padx=5)
        
        run_test_btn = ttk.Button(button_frame, text="Run Batch Test", command=self.run_batch_test)
        run_test_btn.pack(side=tk.LEFT, padx=5)
        
        # Test data info
        ttk.Label(middle_frame, text="Test Data Info:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.test_info_text = tk.Text(middle_frame, height=3, width=100)
        self.test_info_text.pack(fill=tk.X, pady=5)
        
        # ===== BOTTOM SECTION: RESULTS =====
        bottom_frame = ttk.LabelFrame(main_frame, text="Test Results & Metrics", padding=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Results tabs
        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Metrics
        metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(metrics_tab, text="Metrics")
        self.metrics_text = scrolledtext.ScrolledText(metrics_tab, height=15, width=100)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 2: Predictions
        pred_tab = ttk.Frame(self.notebook)
        self.notebook.add(pred_tab, text="Predictions")
        self.predictions_text = scrolledtext.ScrolledText(pred_tab, height=15, width=100)
        self.predictions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 3: Confusion Matrix
        conf_tab = ttk.Frame(self.notebook)
        self.notebook.add(conf_tab, text="Confusion Matrix")
        self.confusion_text = scrolledtext.ScrolledText(conf_tab, height=15, width=100)
        self.confusion_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def auto_train_model(self):
        """Automatically train model from train.txt"""
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Training model from train.txt...\n")
        self.root.update()
        
        thread = threading.Thread(target=self._train_worker)
        thread.start()
    
    def _train_worker(self):
        """Background training worker"""
        try:
            # Load data
            texts = []
            emotions = []
            
            with open('train.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.rsplit(';', 1)
                        if len(parts) == 2:
                            text, emotion = parts
                            texts.append(text)
                            emotions.append(emotion)
            
            self.metrics_text.insert(tk.END, f"Loaded {len(texts)} records\n")
            self.metrics_text.insert(tk.END, "Training model...\n")
            self.root.update()
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                texts, emotions, test_size=0.2, random_state=42, stratify=emotions
            )
            
            # Vectorize
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model = MultinomialNB()
            self.model.fit(X_train_vec, y_train)
            
            # Initial test
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "✓ Model Successfully Trained!\n\n")
            self.metrics_text.insert(tk.END, f"Training set size: {len(X_train)}\n")
            self.metrics_text.insert(tk.END, f"Test set size: {len(X_test)}\n")
            self.metrics_text.insert(tk.END, f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            self.metrics_text.insert(tk.END, "\nReady for testing!\n")
            
        except Exception as e:
            self.metrics_text.insert(tk.END, f"Error: {str(e)}")
    
    def predict_single(self):
        """Predict emotion for single text"""
        if self.model is None or self.vectorizer is None:
            messagebox.showwarning("Warning", "Model not ready yet!")
            return
        
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text!")
            return
        
        try:
            text_vec = self.vectorizer.transform([text])
            emotion = self.model.predict(text_vec)[0]
            probabilities = self.model.predict_proba(text_vec)[0]
            confidence = max(probabilities) * 100
            
            result = f"Emotion: {emotion.upper()} | Confidence: {confidence:.2f}%"
            self.single_result_text.config(text=result, foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def load_test_file(self):
        """Load test data from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                texts = []
                emotions = []
                
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.rsplit(';', 1)
                            if len(parts) == 2:
                                text, emotion = parts
                                texts.append(text)
                                emotions.append(emotion)
                
                self.test_data = texts
                self.emotions = emotions
                
                self.test_info_text.delete(1.0, tk.END)
                self.test_info_text.insert(tk.END, f"✓ Test file loaded: {os.path.basename(filename)}\n")
                self.test_info_text.insert(tk.END, f"Total records: {len(texts)}\n")
                self.test_info_text.insert(tk.END, f"Emotions: {', '.join(set(emotions))}")
                
                messagebox.showinfo("Success", f"Loaded {len(texts)} test records!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def use_train_test_split(self):
        """Use 20% of training data as test set"""
        if self.model is None:
            messagebox.showwarning("Warning", "Model not ready yet!")
            return
        
        try:
            # Load data
            texts = []
            emotions = []
            
            with open('train.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.rsplit(';', 1)
                        if len(parts) == 2:
                            text, emotion = parts
                            texts.append(text)
                            emotions.append(emotion)
            
            # Split
            X_train, self.test_data, y_train, self.emotions = train_test_split(
                texts, emotions, test_size=0.2, random_state=42, stratify=emotions
            )
            
            self.test_info_text.delete(1.0, tk.END)
            self.test_info_text.insert(tk.END, "✓ Using 20% of training data\n")
            self.test_info_text.insert(tk.END, f"Test records: {len(self.test_data)}\n")
            
            emotion_dist = pd.Series(self.emotions).value_counts()
            self.test_info_text.insert(tk.END, "\nEmotion distribution:\n")
            for emotion, count in emotion_dist.items():
                self.test_info_text.insert(tk.END, f"  {emotion}: {count}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def run_batch_test(self):
        """Run batch test on loaded data"""
        if self.model is None or self.vectorizer is None:
            messagebox.showwarning("Warning", "Model not ready!")
            return
        
        if self.test_data is None or self.emotions is None:
            messagebox.showwarning("Warning", "No test data loaded!")
            return
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Running batch test...\n")
        self.root.update()
        
        thread = threading.Thread(target=self._test_worker)
        thread.start()
    
    def _test_worker(self):
        """Background test worker"""
        try:
            # Vectorize and predict
            X_test_vec = self.vectorizer.transform(self.test_data)
            self.predictions = self.model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(self.emotions, self.predictions)
            precision_macro = precision_score(self.emotions, self.predictions, average='macro', zero_division=0)
            recall_macro = recall_score(self.emotions, self.predictions, average='macro', zero_division=0)
            f1_macro = f1_score(self.emotions, self.predictions, average='macro', zero_division=0)
            
            # Display metrics
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "="*60 + "\n")
            self.metrics_text.insert(tk.END, "OVERALL METRICS\n")
            self.metrics_text.insert(tk.END, "="*60 + "\n\n")
            self.metrics_text.insert(tk.END, f"Total test records: {len(self.test_data)}\n")
            self.metrics_text.insert(tk.END, f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            self.metrics_text.insert(tk.END, f"Precision (Macro): {precision_macro:.4f}\n")
            self.metrics_text.insert(tk.END, f"Recall (Macro): {recall_macro:.4f}\n")
            self.metrics_text.insert(tk.END, f"F1-Score (Macro): {f1_macro:.4f}\n\n")
            
            # Per-emotion metrics
            self.metrics_text.insert(tk.END, "="*60 + "\n")
            self.metrics_text.insert(tk.END, "PER-EMOTION METRICS\n")
            self.metrics_text.insert(tk.END, "="*60 + "\n\n")
            
            emotions_list = sorted(set(self.emotions))
            for emotion in emotions_list:
                mask = [e == emotion for e in self.emotions]
                pred_mask = [p == emotion for p in self.predictions]
                
                tp = sum([mask[i] and pred_mask[i] for i in range(len(mask))])
                fp = sum([not mask[i] and pred_mask[i] for i in range(len(mask))])
                fn = sum([mask[i] and not pred_mask[i] for i in range(len(mask))])
                
                emotion_accuracy = sum(mask) / len(mask) * 100 if sum(mask) > 0 else 0
                
                self.metrics_text.insert(tk.END, f"{emotion.upper()}:\n")
                self.metrics_text.insert(tk.END, f"  Count: {sum(mask)}\n")
                self.metrics_text.insert(tk.END, f"  True Positives: {tp}\n")
                self.metrics_text.insert(tk.END, f"  False Positives: {fp}\n")
                self.metrics_text.insert(tk.END, f"  False Negatives: {fn}\n\n")
            
            # Display predictions
            self.predictions_text.delete(1.0, tk.END)
            self.predictions_text.insert(tk.END, "SAMPLE PREDICTIONS (First 20)\n")
            self.predictions_text.insert(tk.END, "="*100 + "\n\n")
            
            for i in range(min(20, len(self.test_data))):
                actual = self.emotions[i]
                predicted = self.predictions[i]
                match = "✓" if actual == predicted else "✗"
                
                self.predictions_text.insert(tk.END, f"{i+1}. {match}\n")
                self.predictions_text.insert(tk.END, f"   Text: {self.test_data[i][:80]}...\n")
                self.predictions_text.insert(tk.END, f"   Actual: {actual} | Predicted: {predicted}\n\n")
            
            # Display confusion matrix
            self.confusion_text.delete(1.0, tk.END)
            self.confusion_text.insert(tk.END, "CONFUSION MATRIX\n")
            self.confusion_text.insert(tk.END, "="*60 + "\n\n")
            
            cm = confusion_matrix(self.emotions, self.predictions, labels=emotions_list)
            
            # Print header
            self.confusion_text.insert(tk.END, "Predicted →\n")
            self.confusion_text.insert(tk.END, "Actual ↓".ljust(15))
            for emotion in emotions_list:
                self.confusion_text.insert(tk.END, emotion.ljust(12))
            self.confusion_text.insert(tk.END, "\n")
            self.confusion_text.insert(tk.END, "-"*70 + "\n")
            
            # Print matrix
            for i, emotion in enumerate(emotions_list):
                self.confusion_text.insert(tk.END, emotion.ljust(15))
                for j in range(len(emotions_list)):
                    self.confusion_text.insert(tk.END, str(cm[i][j]).ljust(12))
                self.confusion_text.insert(tk.END, "\n")
            
        except Exception as e:
            self.metrics_text.insert(tk.END, f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionTestUI(root)
    root.mainloop()
