# bert_sa
sentiment analysis based on [bert](https://github.com/google-research/bert/blob/master/multilingual.md) including training, online predicting and serving with REST 

## Fine tune a sentiment analysis model based on [BERT](https://github.com/google-research/bert)
1. Add a `SAProcessor` and include it within `main` function in run_classifier.py 
2. Prepare train, dev and test files; adapat `_create_examples` method in `SAProcessor` based on your own datasets (pandas may not be required)
3. Specify `BERT_BASE_DIR`, `SA_DIR` and `output_dir` in run_sa.sh and run 


## Test
1. For file based test, change `output_predict_file` in run_classifier.py, specify `TRAINED_CLASSIFIER` and `output_dir` path, run predict_sa.sh 
2. For online prediction, refer to run_classifier_predict_online (modified based on [bert_language_understanding](https://github.com/brightmart/bert_language_understanding))

## Export your model
Refer to sa_predict_saved_model.py

**KIND NOTICE:** some graph definition and input placeholder is imported from run_classifier_predict_online.py

## Serve the model with TensorFlow Serving
1. See [TensorFlow Serving](https://www.tensorflow.org/serving/docker) for details about installing docker and pulling a serving image
2. Running a serving image
```Bash
docker run -p 8501:8501 --name 'bert_sa_serving' --mount type=bind,source=/data/notebooks/xff/bert/output/sa_output/saved_model,target=/models/bert_sa -e MODEL_NAME=bert_sa -t tensorflow/serving:latest-devel-gpu &

docker exec -it bert_sa_serving bash

tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=bert_sa --model_base_path=/models/bert_sa
 ```

3. Sample request
```Python
line=u'建立了完善的质量体系并持续有效运行'
# preprocess is defined in run_classifier_predict_online.py
dict_data = preprocess(line)
resp = requests.post('http://172.17.0.1:8501/v1/models/bert_sa:predict', json=dict_data)
print(resp.json())
```
Results look like this:
{'outputs': {'label_predict': 1, 'possibility': [0.00738544, 0.992615]}}


