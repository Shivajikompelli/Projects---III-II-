import pandas as pd
import string
import nltk
import tkinter as tk
from tkinter import ttk, messagebox
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------- NLTK Setup -------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ----------------- Load and Preprocess Data -------------------
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ----------------- GUI Setup -------------------
def classify_message():
    message = message_entry.get()
    if not message.strip():
        messagebox.showwarning("Input Error", "Please enter a message.")
        return
    processed = preprocess(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    result_label.config(
        text="Prediction: SPAM ❌" if prediction == 1 else "Prediction: HAM ✅",
        foreground="red" if prediction == 1 else "green"
    )

# Create main window
root = tk.Tk()
root.title("SMS Spam Detector")
root.geometry("750x600")
root.resizable(False, False)

# Styling
style = ttk.Style()
style.configure('TLabel', font=('Segoe UI', 20))
style.configure('TButton', font=('Segoe UI', 20), padding=10)

# Layout
ttk.Label(root, text="Enter an SMS message:").pack(pady=30)
message_entry = ttk.Entry(root, width=100)
message_entry.pack()

ttk.Button(root, text="Classify", command=classify_message).pack(pady=20)

result_label = ttk.Label(root, text="", font=("Segoe UI", 16, "bold"))
result_label.pack()

ttk.Label(root, text="Model: MultinomialNB | TF-IDF | NLTK NLP").pack(pady=60)

root.mainloop()
