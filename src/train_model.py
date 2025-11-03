#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/dataset.csv')
parser.add_argument('--model_out', default='models/recommender_rf.pkl')
args = parser.parse_args()

df = pd.read_csv(args.data)
# Features for training
X = df[['cpu_peak_percent','mem_95_mb','cpu_workers','duration_s']]
y = df['recommended_instance']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Train acc:", clf.score(X_train,y_train))
print("Test acc:", clf.score(X_test,y_test))

joblib.dump(clf, args.model_out)
print("Model saved to", args.model_out)
