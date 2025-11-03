#!/usr/bin/env python3
import joblib, pandas as pd, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/recommender_rf.pkl')
parser.add_argument('--metrics', required=True)
args = parser.parse_args()

df = pd.read_csv(args.metrics)
feat = {
    'cpu_peak_percent': df['cpu_percent'].max(),
    'mem_95_mb': df['mem_used_mb'].quantile(0.95),
    'cpu_workers': int(df['cpu_count'].median()),
    'duration_s': len(df)
}
X = pd.DataFrame([feat])
clf = joblib.load(args.model)
pred = clf.predict(X)[0]
print("Predicted instance:", pred)
print("Features used:", X.to_dict(orient='records')[0])
