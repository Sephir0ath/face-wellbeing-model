import pandas as pd


def main(label: str = "depression", mode: str = "Pass"):
    data = pd.read_csv("data.csv")
    data_1 = data[
        (data["question"] == 1)
        & (data["label"] == label)
        & (data["mode"] == mode)
    ]
    data_2 = data[
        (data["question"] == 2)
        & (data["label"] == label)
        & (data["mode"] == mode)
    ]
    data_3 = data[
        (data["question"] == 3)
        & (data["label"] == label)
        & (data["mode"] == mode)
    ]
    data_4 = data[
        (data["question"] == 4)
        & (data["label"] == label)
        & (data["mode"] == mode)
    ]
    data_5 = data[
        (data["question"] == 5)
        & (data["label"] == label)
        & (data["mode"] == mode)
    ]

    my_list = [data_1, data_2, data_3, data_4, data_5]

    for df in my_list:
        df = df[["model", "f1_train", "f1_test"]]
        df = df.sort_values(by="f1_test", ascending=True)
        print(df.head(9), end="\n\n")


if __name__ == "__main__":
    main()
