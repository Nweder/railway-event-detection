import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# === Steg 1: Dataförbehandling ===
df1 = pd.read_csv("tail2.csv")
df2 = pd.read_csv("tail2.csv")
df3 = pd.read_csv("tail3.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.drop(columns=["start_time", "axle", "cluster", "tsne_1", "tsne_2"])
df["event"] = df["event"].apply(lambda x: 0 if x == "normal" else 1)

X = df.drop(columns=["event"])
y = df["event"]

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# === Steg 2: Dela upp i träning/test 80/20 ===
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# === Steg 3: Träna SVM på 80/20 split ===
model_80_20 = SVC()
model_80_20.fit(X_train, y_train)
y_pred = model_80_20.predict(X_test)
accuracy_80_20 = accuracy_score(y_test, y_pred)
print(f"Accuracy med 80/20 split: {accuracy_80_20:.4f}")

# === Steg 4: 5-fold cross-validation ===
model_cv = SVC()
cv_scores = cross_val_score(model_cv, X_normalized, y, cv=5, scoring='accuracy')
print("Cross-validation accuracies:", cv_scores)
print(f"Genomsnittlig CV-accuracy: {cv_scores.mean():.4f}")
