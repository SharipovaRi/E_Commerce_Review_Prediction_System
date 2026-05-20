import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from models.final_pipeline_def import get_pipeline

# Load Data
data_url = "data/raw/train.csv"
df = pd.read_csv(data_url)

X = df.drop(columns=["rating"])
y = df["rating"]

# Fill missing text
X["review_text"] = X["review_text"].fillna("")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load Pipeline
model = get_pipeline()

# Trian Model
model.fit(X_train, y_train)

# Save Final Model 
joblib.dump(model, "models/model.pkl")
print("Model training complete and saved as model.pkl")