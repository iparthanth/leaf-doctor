Filled by tools/install_bundle.py from app_bundle.zip (notebook cell 28.6): labels.txt, app_config.json and the
phone file app_config.json names (tflite_file: model_fp16.tflite, or model_fp32.tflite) - the phone file only if
it passed parity (if not: tools/make_tflite.py). Without it the app still works online through the server;
offline diagnosis is switched off.
