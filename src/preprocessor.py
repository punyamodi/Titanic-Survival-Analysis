import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Lady', 'the Countess', 'Capt', 'Sir', 'Don', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(columns=drop_cols)
    
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    return df

def get_train_test_split(train_path: str):
    train_df = pd.read_csv(train_path)
    y = train_df['Survived']
    X = train_df.drop(columns=['Survived'])
    
    X_cleaned = clean_data(X)
    
    return X_cleaned, y
