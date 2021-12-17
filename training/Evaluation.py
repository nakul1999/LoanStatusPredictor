from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score


class Evaluation:
    def __init__(self, model, val_data):
        self.model = model
        self.xte = val_data[0]
        self.yte = val_data[1]
        self.ypred = self.model.predict(self.xte)



    def precision(self):
        precision = precision_score(self.yte,self.ypred)
        return precision

    def recall(self):
        recall = recall_score(self.yte,self.ypred)
        return recall

    def accuracy(self):
        accuracy = accuracy_score(self.yte,self.ypred)
        return accuracy

    def f1(self):
        f1score = f1_score(self.yte,self.ypred)
        return f1score

    def rocscore(self):
        rocscore = roc_auc_score(self.yte,self.ypred)
        return rocscore

    def evaluateModel(self):

        precision = self.precision()
        recall = self.recall()
        accuracy = self.accuracy()
        f1 = self.f1()
        rocscore = self.rocscore()

        return {"precision": precision,"recall": recall,"accuracy": accuracy,"f1_score": f1,"rocscore": rocscore}
