import onnxruntime as rt
import cv2 
import numpy as np
import time
def facemask_detector(image_array):
    print(image_array.shape)
    session = rt.InferenceSession("service/model.onnx")
    time_init = time.time()
    # Resize the image using cv2
    image = cv2.resize(image_array, (224, 224))
    print(image.shape)
    
    # Transpose and expand dimensions to match the model input format
    image = np.transpose(np.expand_dims(image, axis=0).astype(np.float32), (0, 3, 1, 2))
    print(image.shape)
    
    # Run inference
    onnx_pred = session.run(["Linear"], {"image": image})
    
    # Post-process the output
    res = np.squeeze(onnx_pred).astype(int).item()
    end_time = time.time()
    time_elapsed = end_time - time_init
    print(onnx_pred)
    # Determine if the image has a face mask or not based on the model prediction
    if res < 0.5:
        facemask = "Without Facemask"
    elif res > 0.5:
        facemask = "With Facemask"
    
    return {"facemask": facemask,
            "time_elapsed": str(time_elapsed)}
