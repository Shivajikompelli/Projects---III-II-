import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("udemy_courses.csv")
# df['published_timestamp'] = pd.to_datetime(df['published_timestamp'])
df['published_timestamp'] = pd.to_datetime(df['published_timestamp']).dt.tz_localize(None)
df['course_age_years'] = (pd.Timestamp.now() - df['published_timestamp']).dt.days / 365


df.drop_duplicates(inplace=True)
print("Missing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())
print("\nSubjects:", df['subject'].unique())
print("Levels:", df['level'].unique())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="price", y="num_subscribers", hue="subject")
plt.title("Subscribers vs Price by Subject")
plt.xlabel("Price ($)")
plt.ylabel("Subscribers")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
df.groupby('subject')['content_duration'].mean().sort_values().plot(kind='barh')
plt.title("Average Course Duration (hours) by Subject")
plt.xlabel("Avg Duration")
plt.show()
top_courses = df.sort_values(by="num_subscribers", ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_courses, x="num_subscribers", y="course_title")
plt.title("Top 10 Most Subscribed Courses")
plt.xlabel("Subscribers")
plt.ylabel("Course Title")
plt.xticks(rotation=0)
plt.show()

df['engagement_ratio'] = df['num_reviews'] / df['num_subscribers']
df['engagement_ratio'] = df['engagement_ratio'].fillna(0)

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='level', y='engagement_ratio')
plt.title("Engagement Ratio by Course Level")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='is_paid', y='engagement_ratio')
plt.title("Free vs Paid: Engagement Ratio")
plt.xticks([0, 1], ["Free", "Paid"])
plt.grid(True)
plt.show()

df['is_popular'] = df['num_subscribers'] > df['num_subscribers'].median()
features = df[['price', 'num_lectures', 'content_duration', 'engagement_ratio']]
label = df['is_popular']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Prediction Accuracy:", round(accuracy * 100, 2), "%")

print("\nKey Insights:")
print("- Most subscribed courses are not always the most expensive.")
print("- Intermediate level and 'All Levels' courses show similar engagement.")
print("- Free courses have high variability in engagement.")
print("- Business Finance and Web Development are dominant subjects.")
print("- Prediction model gives a basic idea about factors influencing popularity.")
