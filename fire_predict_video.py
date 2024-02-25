import cv2 as cv
import numpy as np
import tensorflow as tf

labels = {0:'fire', 1:'non_fire'}
model_fire_detection = tf.keras.models.load_model('fire_classification_model')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    resized_frame = cv.resize(frame, (224, 224))

    preprocessed_frame = resized_frame.astype("float32") / 127.5 - 1

    input_frame = np.expand_dims(preprocessed_frame, axis=0)

    predictions = model_fire_detection.predict(input_frame)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_label = labels[predicted_class_index] 

    if predicted_label == 'fire':
        text_color = (0,0,255)
    else:
        text_color = (0,255,0)

    cv.putText(frame, f'Prediction: {predicted_label}', (10,30), cv.FONT_HERSHEY_SIMPLEX, 1,text_color, 2, cv.LINE_AA)

    cv.imshow('Real Time Fire Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
