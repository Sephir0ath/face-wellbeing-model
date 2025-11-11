import pandas as pd

def main(): 
    data = pd.read_csv("data.csv")
    data_1 = data[(data["question"] == 1) & (data["label"] == "anxiety") & (data["mode"] == "selection feature")]
    data_2 = data[(data["question"] == 2) & (data["label"] == "anxiety") & (data["mode"] == "selection feature")]
    data_3 = data[(data["question"] == 3) & (data["label"] == "anxiety") & (data["mode"] == "selection feature")]
    data_4 = data[(data["question"] == 4) & (data["label"] == "anxiety") & (data["mode"] == "selection feature")]
    data_5 = data[(data["question"] == 5) & (data["label"] == "anxiety") & (data["mode"] == "selection feature")]

    my_list = [data_1, data_2, data_3, data_4, data_5]

    for df in my_list:
        df = df[["model", "f1_train", "f1_test"]]
        df = df.sort_values(by="f1_test", ascending=True)
        print(df.head(9), end="\n\n")


if __name__ == "__main__":
    main()