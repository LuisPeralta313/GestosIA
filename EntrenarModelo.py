import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Crear y entrenar el modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Realizar predicciones
y_predict = model.predict(x_test)

# Calcular precisión
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Generar informe de métricas (precisión, recall, F1-score)
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_predict))

# Generar y visualizar matriz de confusión
cm = confusion_matrix(y_test, y_predict, labels=np.unique(labels))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.savefig('confusion_matrix.png')  # Guardar la matriz como imagen
plt.close()

# Guardar el modelo
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()