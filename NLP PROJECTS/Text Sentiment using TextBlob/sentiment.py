import tkinter as tk
from tkinter import ttk, messagebox
from textblob import TextBlob

# ---------- Sentiment Analysis Logic ----------
def analyze_sentiment():
    text = text_entry.get()
    if not text.strip():
        messagebox.showwarning("Input Error", "Please enter some text.")
        return

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Classification logic
    if polarity > 0:
        sentiment = "Positive"
        color = "green"
    elif polarity < 0:
        sentiment = "Negative"
        color = "red"
    else:
        sentiment = "Neutral"
        color = "blue"

    result_label.config(text=f"Sentiment: {sentiment}", foreground=color)

# ---------- GUI Setup ----------
root = tk.Tk()
root.title("Text Sentiment Analyzer")
root.geometry("500x300")
root.resizable(False, False)

# Styling
style = ttk.Style()
style.configure('TLabel', font=('Segoe UI', 11))
style.configure('TButton', font=('Segoe UI', 11), padding=6)

# Layout
ttk.Label(root, text="Enter your text for sentiment analysis:").pack(pady=15)
text_entry = ttk.Entry(root, width=60)
text_entry.pack(pady=5)

ttk.Button(root, text="Analyze", command=analyze_sentiment).pack(pady=20)

result_label = ttk.Label(root, text="", font=("Segoe UI", 13, "bold"))
result_label.pack()

ttk.Label(root, text="Model: TextBlob | Rule-Based NLP").pack(pady=15)

root.mainloop()



