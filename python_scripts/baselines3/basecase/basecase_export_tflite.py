from baselines3.sb3_to_tflite import sb3_to_tensorflow, tf_to_tflite, sb3_to_tflite_pipeline


if __name__ == '__main__':
    sb3_to_tflite_pipeline("stable-model-best/best_model")
