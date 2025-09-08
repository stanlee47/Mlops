import sklearn as sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import yaml

# Load config from params.yaml
config = yaml.safe_load(open('params.yaml'))['preprocess']
input_path = config['input']
output_path = config['output']

df=pd.read_csv(input_path)

label_encoder=LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column]=label_encoder.fit_transform(df[column])
scaler=StandardScaler()
numerical_cols=df.select_dtypes(include=['int64','float64']).columns
df[numerical_cols]=scaler.fit_transform(df[numerical_cols])
df.to_csv(output_path,index=False)
print(f"Preprocessed data savedd to {output_path}")
