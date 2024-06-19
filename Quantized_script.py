import numpy as np
import tensorflow as tf
import cv2 as cv


def preprocess_image(img, input_size=244):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = img.shape
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized_img = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_img = np.zeros((input_size, input_size, 1), dtype=np.uint8)
    new_img[0:new_height, 0:new_width, 0] = resized_img
    new_img = np.expand_dims(new_img, axis=0)
    return new_img



interpreter = tf.lite.Interpreter(model_path='tflite_models/tflite_model_quant.tflite')
interpreter.allocate_tensors()
input_tensor_index = interpreter.get_input_details()[0]['index']
output = interpreter.get_output_details()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    original_width, original_height = frame.shape[:2]
    preprocessed_frame = preprocess_image(frame)
    interpreter.set_tensor(input_tensor_index, preprocessed_frame)
    interpreter.invoke()

    bounding_box = interpreter.get_tensor(output[0]['index'])
    class_probability = np.argmax(interpreter.get_tensor(output[1]['index'])[0])

    x, y, w, h = bounding_box[0]
    x = x*original_width
    y = y*original_height
    h = h*original_height
    w = w*original_width
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if class_probability==0:
        class_text = 'Class: Masked'
    else:
        class_text = 'Class: Unmasked'
    cv.putText(frame, class_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Object Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()