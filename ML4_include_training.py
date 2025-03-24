import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping

def load_data(path):
    train_dataset = image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        label_mode="categorical",
        seed=0,
        image_size=(100, 100),
    )
    test_dataset = image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=0,
        image_size=(100, 100),
    )
    return train_dataset, test_dataset

def train_model_with_logging(model, train_dataset, test_dataset, epochs, output_file):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        verbose=1,
        callbacks=[early_stopping]
    )
    epoch_range = range(1, len(history.history['accuracy']) + 1)
    training_acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']

    accuracy_table = pd.DataFrame({
        'Epoch': epoch_range,
        'Training Accuracy': training_acc,
        'Validation Accuracy': validation_acc
    })
    print("\nTraining Accuracy Over Epochs:\n")
    print(accuracy_table)
    accuracy_table.to_csv(output_file, index=False)
    return history

def cnn_model1():
    model = Sequential([
        Input((100, 100, 3)),
        Rescaling(1/255),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(6, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def cnn_model2():
    model = Sequential([
        Input((100, 100, 3)),
        Rescaling(1/255),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(6, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def fine_tune_model():
    base_model = VGG16(include_top=False, input_shape=(100, 100, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(6, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def test_custom_images(image_dir, model1, model2, correct_classes):
    results = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, filename)
            img = load_img(img_path, target_size=(100, 100))
            img_array = img_to_array(img) / 255.0
            img_array = img_array.reshape(1, 100, 100, 3)

            prediction1 = model1.predict(img_array)
            predicted_class1 = np.argmax(prediction1)

            prediction2 = model2.predict(img_array)
            predicted_class2 = np.argmax(prediction2)

            results.append({
                "Filename": filename,
                "Correct Class": correct_classes.get(filename, "Unknown"),
                "Task 1 Prediction": predicted_class1,
                "Fine-tuned Prediction": predicted_class2
            })

            plt.imshow(img)
            plt.title(f"Correct: {correct_classes.get(filename, 'Unknown')}\n"
                      f"Task 1: {predicted_class1}, Fine-tuned: {predicted_class2}")
            plt.show()

    results_df = pd.DataFrame(results)
    results_df.to_csv("prediction_results.csv", index=False)
    print("\nResults saved to 'prediction_results.csv'.")

def save_test_accuracy_table(test_accuracies):
    test_accuracy_table = pd.DataFrame(test_accuracies)
    print("\nTest Accuracy Table:")
    print(test_accuracy_table)
    test_accuracy_table.to_csv("test_accuracy_comparison.csv", index=False)

def main():
	
    path = "./6 Emotions for image classification"
    for file in glob.glob(os.path.join(path, "**/*"), recursive=True):
        if os.path.isfile(file):
            try:
                with open(file, "rb") as f:
                # 使用 f.read(10) 代替 f.peek(10) 以避免不必要的檔案佔用
                    if not b"JFIF" in f.read(10):
                        f.close()  # 確保檔案已關閉後再刪除
                        os.remove(file)
            except PermissionError:
                print(f"PermissionError: Unable to process {file}")

    train_dataset, test_dataset = load_data(path)

    print("\nTraining Model 1 (Without dropout and pooling):")
    model1 = cnn_model1()
    train_model_with_logging(model1, train_dataset, test_dataset, epochs=50, output_file="model1_accuracy.csv")
    model1.save("cnn_model1.keras")
    test_score1 = model1.evaluate(test_dataset, verbose=2)
    print(f"\nModel 1 Test Accuracy: {test_score1[1]:.3f}")

    print("\nTraining Model 2 (With dropout and pooling):")
    model2 = cnn_model2()
    train_model_with_logging(model2, train_dataset, test_dataset, epochs=50, output_file="model2_accuracy.csv")
    model2.save("cnn_model2.keras")
    test_score2 = model2.evaluate(test_dataset, verbose=2)
    print(f"\nModel 2 Test Accuracy: {test_score2[1]:.3f}")

    better_model = model1 if test_score1[1] > test_score2[1] else model2

    print("\nFine-tuning VGG16:")
    fine_tuned_model = fine_tune_model()
    train_model_with_logging(fine_tuned_model, train_dataset, test_dataset, epochs=50, output_file="fine_tuned_accuracy.csv")
    fine_tuned_model.save("fine_tuned_model.keras")
    test_score_ft = fine_tuned_model.evaluate(test_dataset, verbose=2)
    print(f"\nFine-tuned Model Test Accuracy: {test_score_ft[1]:.3f}")
    
    # Save test accuracy table
    test_accuracies = {
        "Model": ["Model 1 (No dropout, no pooling)", 
                  "Model 2 (With dropout and pooling)", 
                  "Fine-tuned VGG16"],
        "Test Accuracy": [test_score1[1], test_score2[1], test_score_ft[1]]
    }
    save_test_accuracy_table(test_accuracies)

    test_custom_images(
        image_dir="./emotion",
        model1=better_model,
        model2=fine_tuned_model,
        correct_classes={
            "image1.jpg": 1, "image2.jpg": 1, "image3.jpg": 2,
            "image4.jpg": 2, "image5.jpg": 3, "image6.jpg": 3,
            "image7.jpg": 4, "image8.jpg": 4, "image9.jpg": 5,
            "image10.jpg": 6
        }
    )


main()
