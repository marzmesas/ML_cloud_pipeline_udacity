# ML_cloud_pipeline_udacity

This project contains the development of a classification model on Census Bureau data. 
The main goal is to robustly deploy a machine learning model into production.  
This includes: 
* preprocessing the data and training the model
* testing the code using pytest
* deploying the model using the FastAPI package and creating API tests on Render
* incorporating the ML pipeline into a CI/CD framework using GitHub Actions.

### Model  

* To train the model run:
``` 
python model/ml/train_model.py
```

### Render deployment  

* Alternatively test the model live on Heroku by executing a POST request:

```
python api_client.py
```

### GitHub Actions  

The machine learning pipeline is deployed automatically in a CI/CD fashion. After successfully passing the tests, the code is automaticaly pushed to the Render instance.
