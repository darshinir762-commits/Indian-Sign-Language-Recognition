from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import test_dataset, class_names

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR.parent / "backend" / "saved_models"


def evaluate_model(model_name, model_file):
    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    model_path = SAVE_DIR / model_file

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Basic evaluation
    loss, accuracy = model.evaluate(test_dataset, verbose=1)

    print(f"\n{model_name} Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} Test Loss: {loss:.4f}")

    # Predictions
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

        # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix as image
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    plt.colorbar()

    plt.tight_layout()

    REPORT_DIR = BASE_DIR.parent.parent / "reports" / "confusion_matrices"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        REPORT_DIR / f"{model_name.lower()}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Confusion matrix saved for {model_name}")

# Evaluate models
evaluate_model("CNN", "cnn_model.keras")
evaluate_model("MobileNetV2", "mobilenet_model.keras")
evaluate_model("EfficientNetB0", "efficientnet_model.keras")