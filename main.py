from src.data_extractor import DataExtractor


def main():
    # Extract csv data for temporality, question and feature
    extractor = DataExtractor()
    dataframe = extractor.extract_csv(
        temporality=True, question=2, feature="gaze", labels="depression"
    )

    print(dataframe.head())


if __name__ == "__main__":
    main()
