from predict import predict

if __name__ == "__main__":
    image = "cat.jpg"
    prediction = predict(image)
    print(f"Prediction: {prediction}")