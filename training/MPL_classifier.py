import numpy as np
import os,time
import random
import librosa  # still included for consistency
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.utils import shuffle

# ----------------------
# Paths
# ----------------------
dataset_path = r"/home/ranjit/Desktop/projects/Hello_Nisha/HuBERT_dataset/hubert_embeddings.npz"
results_txt = r"/home/ranjit/Desktop/projects/Hello_Nisha/MLP_results.txt"
model_dir = r"/home/ranjit/Desktop/projects/Hello_Nisha/Model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "mlp_model.pkl")
best_model_path = os.path.join(model_dir, "mlp_model_best.pkl")

# ----------------------
# Load dataset
# ----------------------
data = np.load(dataset_path)
embeddings = data["embeddings"]
labels = data["labels"]

print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape: {labels.shape}")

# ----------------------
# Split data
# ----------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ----------------------
# Augmentation function
# ----------------------
def augment_embeddings(batch, noise_std=0.01, dropout_rate=0.05, mask_fraction=0.1):
    augmented = batch.copy()

    # Add Gaussian noise with varied std
    noise = np.random.normal(0, noise_std * np.random.uniform(0.5, 1.5), augmented.shape)
    augmented += noise

    # Apply random dropout
    dropout_mask = np.random.rand(*augmented.shape) > dropout_rate
    augmented *= dropout_mask

    # Mask random dimensions
    num_features = augmented.shape[1]
    num_mask = int(mask_fraction * num_features)
    if num_mask > 0:
        for row in augmented:
            mask_indices = random.sample(range(num_features), num_mask)
            row[mask_indices] = 0.0

    return augmented

# ----------------------
# Define MLP classifier
# ----------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=1,  # we will use partial_fit loop
    warm_start=True,
    random_state=42,
    verbose=False
)

# Initialize partial_fit with classes
classes = np.unique(y_train)
mlp.partial_fit(X_train[:2], y_train[:2], classes=classes)

# ----------------------
# Training loop with early stopping
# ----------------------
epochs = 50
patience = 5
best_val_acc = 0.0
patience_counter = 0
history = {"train_acc": [], "val_acc": []}

for epoch in range(epochs):
    # Shuffle training data each epoch
    X_train, y_train = shuffle(X_train, y_train, random_state=epoch)
    X_train_aug = augment_embeddings(X_train)

    # Train with partial_fit
    mlp.partial_fit(X_train_aug, y_train)

    # Evaluate on train and validation
    y_train_pred = mlp.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_val_pred = mlp.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        joblib.dump(mlp, best_model_path)
        print(f"✅ Best model updated at epoch {epoch+1} with Val Acc {val_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping triggered")
            break

# ----------------------
# Final evaluation (using best model)
# ----------------------
best_mlp = joblib.load(best_model_path)
y_pred = best_mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc = roc_auc_score(y_test, best_mlp.predict_proba(X_test)[:, 1])

print("=== MLP Final Evaluation ===")
print(f"Accuracy: {acc}")
print(f"ROC-AUC: {roc:.4f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

# ----------------------
# Save results to txt
# ----------------------
with open(results_txt, "w") as f:
    f.write("=== MLP Final Evaluation ===\n")
    f.write(f"Accuracy: {acc}\n")
    f.write(f"ROC-AUC: {roc:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\nClassification Report:\n")
    f.write(report)
    f.write("\nTraining History (Accuracies):\n")
    for e, (ta, va) in enumerate(zip(history["train_acc"], history["val_acc"])):
        f.write(f"Epoch {e+1}: Train {ta:.4f}, Val {va:.4f}\n")

print(f"Results saved to {results_txt}")

# ----------------------
# Save final trained model
# ----------------------
print("Sleeping for 30 seconds")
time.sleep(30)
joblib.dump(best_mlp, model_path)
print(f"Final trained MLP model saved to {model_path}")
