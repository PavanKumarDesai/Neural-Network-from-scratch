import pandas as pd
import numpy as np
def pre_processing(df):
    #cleaning the dataset
    df['Age']=df['Age'].fillna(df['Age'].mean())
    df.Age = df.Age.round()
    df['Weight'] = df['Weight'].fillna(df.groupby('Age')['Weight'].transform('mean'))
    df['Delivery phase']=df['Delivery phase'].fillna(df['Delivery phase'].mode()[0])
    df['HB'] = df['HB'].fillna(df.groupby('Age')['HB'].transform('mean'))
    df['BP'] = df['BP'].fillna(df.groupby('Age')['BP'].transform('mean'))
    df['Education']=df['Education'].fillna(df['Education'].mode()[0])
    df['Residence']=df['Residence'].fillna(df['Residence'].mode()[0])
    df.Weight = df.Weight.round()
    df.HB = df.HB.round()
    df=df.apply(lambda x: x.fillna(x.mean()),axis=0).round()
    #standardizing the dataset
    df['Age']=(df['Age']-df['Age'].mean())/(df['Age'].std())
    df['Weight']=(df['Weight']-df['Weight'].mean())/(df['Weight'].std())
    df['HB']=(df['HB']-df['HB'].mean())/(df['HB'].std())
    df['BP']=(df['BP']-df['BP'].mean())/(df['BP'].std())
    #one-hot-encoding for categorical data
    y = pd.get_dummies(df.Community, prefix='Community')
    df['Community_1']=y['Community_1']
    df['Community_2']=y['Community_2']
    df['Community_3']=y['Community_3']
    df['Community_4']=y['Community_4']
    df=df.drop(['Community'], axis = 1)
    temp=df['Result']
    df=df.drop(['Result'], axis = 1)
    df['Delivery phase'] = df['Delivery phase'].map({1: 0, 2: 1})
    df['Result']=temp
    return df

if __name__=="__main__":
	df=pd.read_csv("LBW_Dataset.csv")
	df=pre_processing(df)
	df.to_csv("../data/pre_processed.csv",index=False)