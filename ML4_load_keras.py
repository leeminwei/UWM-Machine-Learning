import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

def evaluate_models_on_test_data(test_dataset, models):
    results = []
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} on Test Dataset...")
        test_score = model.evaluate(test_dataset, verbose=2)
        print(f"{model_name} Test Accuracy: {test_score[1]:.3f}")
        results.append({"Model": model_name, "Test Accuracy": test_score[1]})
    return results

def test_emotion_images(image_dir, fine_tuned_model, correct_classes):
    print("\nTesting Fine-tuned Model on Emotion Dataset...")
    results = []

    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, filename)

            # Load and preprocess image
            img = load_img(img_path, target_size=(100, 100))
            img_array = img_to_array(img) / 255.0
            img_array = img_array.reshape(1, 100, 100, 3)

            # Predict
            prediction_fine_tuned = fine_tuned_model.predict(img_array, verbose=0)
            predicted_class_fine_tuned = np.argmax(prediction_fine_tuned)

            # Save result
            results.append({
                "Filename": filename,
                "Correct Class": correct_classes.get(filename, "Unknown"),
                "Fine-tuned Model Prediction": predicted_class_fine_tuned
            })

            # Display image and predictions
            plt.imshow(img)
            plt.title(f"Correct: {correct_classes.get(filename, 'Unknown')}\n"
                      f"Fine-tuned Prediction: {predicted_class_fine_tuned}")
            plt.axis('off')
            plt.show()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("emotion_results.csv", index=False)
    print("\nEmotion Dataset Results saved to 'emotion_results.csv'.")

def main():
    # Load models
    model1_path = "./cnn_model1.keras"
    model2_path = "./cnn_model2.keras"
    fine_tuned_path = "./fine_tuned_model.keras"

    print("Loading Model 1...")
    model1 = load_model(model1_path)

    print("Loading Model 2...")
    model2 = load_model(model2_path)

    models = {"Model 1": model1, "Model 2": model2}

    fine_tuned_model = None
    if os.path.exists(fine_tuned_path):
        print("Loading Fine-tuned Model...")
        fine_tuned_model = load_model(fine_tuned_path)
        models["Fine-tuned Model"] = fine_tuned_model

    # Load test dataset
    print("Loading Test Dataset...")
    test_dataset = image_dataset_from_directory(
        "./6 Emotions for image classification",
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=0,
        image_size=(100, 100),
    )

    # Evaluate models on test data
    test_results = evaluate_models_on_test_data(test_dataset, models)

    # Save test accuracy results to CSV
    test_accuracy_df = pd.DataFrame(test_results)
    test_accuracy_df.to_csv("test_accuracy_comparison.csv", index=False)
    print("\nTest Accuracy Table saved to 'test_accuracy_comparison.csv'.")
    print(test_accuracy_df)

    # Test emotion dataset only if fine-tuned model exists
    if fine_tuned_model:
        emotion_dir = "./emotion"
        correct_classes = {
            "image1.jpg": 1, "image2.jpg": 1, "image3.jpg": 2,
            "image4.jpg": 2, "image5.jpg": 3, "image6.jpg": 3,
            "image7.jpg": 4, "image8.jpg": 4, "image9.jpg": 5,
            "image10.jpg": 6
        }
        test_emotion_images(emotion_dir, fine_tuned_model, correct_classes)

main()
