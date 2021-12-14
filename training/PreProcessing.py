import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class PreProcessing:
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath)
        print("preprocessing.. ")

    def drop_LOANID(self):
        print(self.data.columns)
        if "Loan_ID" in self.data.columns:
            print("dropping loan_id")
            self.data = self.data.drop(columns=['Loan_ID'], axis=1)
            print(self.data.head(5))
        else:
            print("loanid dropped")

        return self.data

    def preprocessing(self):
        print("preprocessing begins ....")
        CAT_COLS = []
        NUM_COLS = []
        COLUMNS = self.data.columns
        for column in COLUMNS:
            if self.data[column].dtype == 'object' or column in ['Credit_History', "Loan_Amount_Term"]:
                CAT_COLS.append(column)
            else:
                NUM_COLS.append(column)

        print(NUM_COLS)
        print(CAT_COLS)
        le = LabelEncoder()
        for cat_col in CAT_COLS:
            if self.data[cat_col].isna().any():
                most_feq = self.data[cat_col].mode(dropna=True)
                print(most_feq)
                self.data[cat_col].fillna(most_feq, inplace=True)

            self.data[cat_col] = le.fit_transform(self.data[cat_col])

        mm = MinMaxScaler()
        for num_col in NUM_COLS:
            if self.data[num_col].isna().any():
                median = self.data[num_col].median()
                print(median)
                self.data[num_col].fillna(median, inplace=True)

            self.data[[num_col]] = pd.DataFrame(mm.fit_transform(self.data[[num_col]]))

        return self.data

    def split_data(self,data):
        x, y = data.iloc[:, :-1], data.iloc[:, -1]
        return train_test_split(x,y,test_size=0.25,stratify=y)


