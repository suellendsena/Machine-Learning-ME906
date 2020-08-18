
library(tidyverse)
library(caret)
library(ggplot2)
library(GGally)
library(e1071)


## Algoritmos de Classificação: *Naive Bayes*, KNN e Regressão Logística

## Objetivo

# implementar esses três algoritmos no conjunto de dados de mulheres indianas 
# (PIMA Indians Diabetes Database) para diagnosticar se essas pacientes têm ou não diabetes 
# baseando-se em algumas medidas de diagnóstico.

## Dados

# Será trabalho o conjunto de dados PIMA Indians Diabetes Database disponível no Kaggle: 
# https://www.kaggle.com/uciml/pima-indians-diabetes-database 

# Este conjunto de dados é original do Instituto Nacional de Diabetes e Doenças Digestivas e Renais. 
# O objetivo é prever se um paciente tem ou não diabetes, com base em certas medidas de diagnóstico 
# incluídas no conjunto de dados. Várias restrições foram colocadas na seleção dessas pacientes. 
# Em particular, todos os pacientes aqui são mulheres com pelo menos 21 anos de idade da herança 
# indígena Pima.

# Dentre as variáveis preditoras temos o número de gestações que a paciente teve, seu IMC, 
# nível de insulina, idade e algumas outras medidas.


## Análise exploratória inicial

bd = read.csv("diabetes.csv")
bd$Outcome = as.factor(bd$Outcome)
ggpairs(bd, colour="Outcome")
summary(bd)

## Separando em treino e teste 

set.seed(177261)
aux = createDataPartition(bd$Outcome, p = 0.8, list = FALSE)

bdtrain = bd[aux,]
bdtest = bd[-aux,]

Xtrain = bdtrain[,-9]
ytrain = bdtrain$Outcome
Xtest = bdtest[,-9]
ytest = bdtest$Outcome

protreino = preProcess(Xtrain, method = c("center", "scale"))



## Naive Bayes 

bdtrain$Outcome = as.factor(bdtrain$Outcome)
bdtest$Outcome = as.factor(bdtest$Outcome)
bayesfit = naiveBayes(Outcome ~ ., data = bdtrain)  

predtest = predict(bayesfit, newdata = bdtest)

matrix1 = confusionMatrix(predtest, bdtest$Outcome)
matrix1



grid = expand.grid(Glucose = 0:200, Age = 20:90)
predgrid = predict(bayesfit, grid)

grid = cbind(grid, predgrid)
aux = c("Glucose", "Age", "Outcome")
colnames(grid) = aux

ggplot(data = bd, aes(x = Glucose, y = Age, fill = Outcome)) + 
  geom_point(data = grid, aes(x = Glucose, y = Age, colour = Outcome), size = 0000000.1, alpha = 0.50) +
  geom_point(data = bdtrain, shape = 16, aes(colour = Outcome)) +
  geom_point(data = bdtest, shape = 23, aes(fill = Outcome)) 


## KNN por validação cruzada

set.seed(177261)

bdtrain1 = bdtrain[,c(2, 8, 9)]
bdtest1 = bdtest[,c(2, 8, 9)]

Xtrain = bdtrain1[,-9]
ytrain = bdtrain1$Outcome
Xtest = bdtest1[,-9]
ytest = bdtest1$Outcome

protreino = preProcess(Xtrain, method = c("center", "scale"))

train.control = trainControl(method = "cv", number = 15)

kgrid = data.frame(k = c(1, seq(1, 100, by = 2)))

knn.fit = train(Outcome ~ ., 
                data = bdtrain1,
                method = "knn",
                trControl = train.control,
                tuneGrid = kgrid)

knn.fit = train(Outcome ~ ., 
                data = bdtrain1,
                method = "knn",
                preProcess = c("center", "scale"),
                trControl = train.control,
                tuneGrid = knn.fit$bestTune)


predtest2 = predict(knn.fit, newdata = bdtest1)

matrix2 = confusionMatrix(predtest2, bdtest$Outcome)
matrix2


## KNN por k ótimo 

grid = expand.grid(Glucose = 0:200, Age = 20:90)
predgrid = predict(knn.fit, grid)

grid = cbind(grid, predgrid)
colnames(grid) = c("Glucose", "Age", "Outcome")


ggplot(data = bd, aes(x = Glucose, y = Age, fill = Outcome)) + 
  geom_point(data = grid, aes(x = Glucose, y = Age, colour = Outcome), size = 0000000.1, alpha = 0.50) +
  geom_point(data = bdtrain, shape = 16, aes(colour = Outcome)) +
  geom_point(data = bdtest, shape = 23, aes(fill = Outcome)) 



## Regressão Logística e matriz de confusão 

logist.fit = glm(Outcome ~ ., family = "binomial", data = bdtrain)

predtest3 = predict(logist.fit, newdata = bdtest, type = "response")
class.lm = ifelse(predtest3 >= 0.5, 1, 0)

matrix3 = confusionMatrix(as.factor(class.lm), bdtest$Outcome)
matrix3



## Compare os três algoritmos. 


data.frame(Modelo = c("Nayve Bayes", "KNN", "Logístico"),
           "Acurácia" = c(matrix1$overall[[1]], matrix2$overall[[1]], matrix3$overall[[1]]),
           "Sensibilidade" = c(matrix1$byClass[[1]], matrix2$byClass[[1]], matrix3$byClass[[1]]),
           "Especificidade" = c(matrix1$byClass[[2]], matrix2$byClass[[2]], matrix3$byClass[[2]]))

#matrix1$table
#matrix2$table
#matrix3$table


# A partir dos resultados obtidos o modelo de regressão KNN apresentou os piores resultados 
# quando comparado a Nayve Bayes e Regressão Logística. Entre os dois últimos, têm-se que o melhor 
# ajuste para os dados é utilizando Nayve Bayes.

