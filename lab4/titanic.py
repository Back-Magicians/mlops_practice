from catboost.datasets import titanic


FILENAME = "titanic_train.csv"
REQUIRED_FIELDS = ["Pclass", "Sex", "Age"]

titanic_train, _ = titanic()


def main():
    titanic_train.to_csv(FILENAME, columns=REQUIRED_FIELDS)


if __name__ == "__main__":
    main()
