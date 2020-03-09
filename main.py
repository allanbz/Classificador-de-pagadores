import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Lendo banco de dados pós-tratamento
df_emprestimos_banco = pd.read_csv('df_emprestimos_banco_final.csv')

# Separando variáveis de entrada e saída do modelo
df_output = df_emprestimos_banco['PAGO']
df_input = df_emprestimos_banco.drop('PAGO', 1)

# Estruturando dados em array
df_input = df_input.values
df_output = df_output.values

# Separando dados em sets de treino e teste (80% e 20% dos dados, respectivamente)
input_train, input_test, output_train, output_test = train_test_split(df_input, df_output, test_size = 0.2, random_state = 42)

# Declarando modelo
model = Sequential()
model.add(Dense(32, input_shape = (input_train.shape[1], ), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# Compilando modelo
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Treinando modelo
model.fit(input_train, output_train, batch_size = 32, epochs = 100)

# Avaliando modelo
score = model.evaluate(input_test, output_test)

# Exibindo acurácia do modelo
print('Acurácia:', score[1])