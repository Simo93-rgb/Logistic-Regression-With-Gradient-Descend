# Carica e pre-processa i dati
from funzioni import *

X, y = carica_dati()

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
	X,
	y,
	test_size=0.3,
	random_state=42,
	stratify=y
)

X_train, y_train, preprocess_artifacts = fit_preprocess_train(
	X_train_raw,
	y_train_raw,
	class_balancer="",
	corr=0.95
)

X_test, y_test = transform_with_fitted_preprocess(
	X_test_raw,
	y_test_raw,
	preprocess_artifacts
)

# Salva un dataset preprocessato in modo leak-safe (fit su train, apply su test)
file_name = 'breast_cancer_wisconsin_edited'
file_path = 'Assets/dataset'

df_train = pd.DataFrame(X_train, columns=preprocess_artifacts['remaining_feature_names'])
df_train['target'] = y_train

df_test = pd.DataFrame(X_test, columns=preprocess_artifacts['remaining_feature_names'])
df_test['target'] = y_test

df = pd.concat([df_train, df_test], ignore_index=True)
csv_file = os.path.join(file_path, f'{file_name}.csv')
df.to_csv(csv_file, index=False)
print(f"Dataset leak-safe salvato in {csv_file}")
