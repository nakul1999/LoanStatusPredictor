from fastapi import FastAPI
from Config import Config
from FileOperation import FileOperation
from training.PreProcessing import PreProcessing
from training.Training import Training

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/training")
async def preprocess():
    config = Config()
    datapath = config.trainingDataPath
    training = Training(datapath)
    xtr, xte, ytr, yte = training.preprocessing()
    return training.model_training(xtr, xte, ytr, yte)


@app.get("/prediction")
async def predict():
    config = Config()
    modelpath = config.modelSavePath
    fileOps = FileOperation()
    model = fileOps.loadModel("dtc_1", modelpath)
    ypred = model.predict(yte)
    return str(ypred)
