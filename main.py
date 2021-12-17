from fastapi import FastAPI
from Config import Config
from FileOperation import FileOperation
from training.Evaluation import Evaluation
from training.PreProcessing import PreProcessing
from training.Training import Training

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/preprocess")
async def preprocess():
    config = Config()
    datapath = config.trainingDataPath
    # training = Training(datapath)
    preprocessing = PreProcessing(datapath)
    modelSaved = preprocessing.preprocess()
    # modelSaved = training.preprocessing()
    if (modelSaved):
        print("DATA saved successfully")
    else:
        print("DATA not saved")
    return modelSaved


@app.get("/training")
async def trainModel():
    training = Training()
    modelSaved = training.trainModel()
    return modelSaved


@app.get("/modelperformance")
async def preprocess():
    config = Config()
    fileops = FileOperation()
    modelPath = config.modelSavePath
    savedDataPath = config.preprocesseddatapath
    model = fileops.loadModel("model_1", modelPath)
    savedData = fileops.loadModel("preprocessed_data", savedDataPath)
    valData = [savedData[1], savedData[3]]
    evaluation = Evaluation(model, valData)
    score = evaluation.evaluateModel()
    return score

# @app.get("/prediction")
# async def predict():
#     # config = Config()
#     # modelpath = config.modelSavePath
#     # fileOps = FileOperation()
#     # model = fileOps.loadModel("dtc_1", modelpath)
#     # ypred = model.predict(yte)
#     # return str(ypred)
