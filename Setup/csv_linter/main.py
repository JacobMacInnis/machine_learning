import pandas as pd


def main():
    df = pd.DataFrame({
        'name': ['Jacob', 'Kasia', 'Oliver'],
        'age': [23, 25, 22],
    })

    print("✅ DataFrame loaded successfully:\n")
    print(df)
