import matplotlib.pyplot as plt
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Model results
models = ["CNN", "MobileNetV2", "EfficientNetB0"]
accuracies = [99.91, 99.98, 99.94]

# Create graph
plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Deep Learning Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(99, 100)

# Display values above bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        accuracy + 0.02,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

# Save graph
output_path = PROJECT_ROOT / "ai" / "reports" / "images" / "model_accuracy_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Graph saved successfully: {output_path}")

plt.show()