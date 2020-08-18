library(tidyverse)
library(ISLR)
library(caret)
library(rpart)
library(rpart.plot)
library(data.table)
library(ModelMetrics)

## Árvores de Regressão e Classificação

## Objetivo

# Implementar árovres de regressão e classificação em dois conjuntos de dados. 
# O primeiro são os dados `Hitters` do pacote `ISLR` e o segundo é o PIMA Indians Diabetes Database, 
# ambos já utilizados em laboratórios anteriores com outros algoritmos de regressão e classificação, respectivamente. 


### Parte 1: Árvore de Regressão nos dados `Hitters`



data("Hitters")

## Removendo dados faltantes
Hitters = na.omit(Hitters)

##  Separando dados
set.seed(177261)

indices = createDataPartition(Hitters$Salary, times = 1, p = 0.8, list = F)
treino = Hitters[indices,]
teste = Hitters[-indices,]


ggplot(Hitters, aes(Years, Hits, color = Salary)) + geom_point() + scale_color_gradient2(midpoint= mean(Hitters$Salary), low="blue", mid="purple",high="red", space ="Lab")



## Ajuste uma árvore de regressão sem poda usando apenas os dois preditores `Years` e `Hits`. 
# Dica: use o pacote `rpart`. 

fittree = rpart(Salary ~ Years + Hits, data = treino)


## Gráfico da árvore resultante usando a função `rpart.plot()` 


rpart.plot(x = fittree, main = "Árvore de Decisão", type = 1)
rpart.plot(x = fittree, main = "Árvore de Decisão", type = 5, extra = 100)


#Calculando os MSE

MSEtrain = mse(predict(fittree), treino$Salary)
MSEtest = mse(predict(fittree, teste), teste$Salary)

#Tabela com os MSE's

data.table(MSEtrain, MSEtest)



## Árvore de regressão sem poda. 

fitcompleto = rpart(Salary ~ ., data = treino)
MSETrainCompleto = mse(predict(fitcompleto), treino$Salary)
MSETestCompleto = mse(predict(fitcompleto, teste), teste$Salary)

# MSE
data.table(MSETrainCompleto, MSETestCompleto)


# O parâmetro de ajuste fino é o tamanho da árvore que é determinado pelo parâmetro de complexidade (cp). 
# Escolha do valor desse parâmetro por validação cruzada e gráfico da árvore resultante. 
# Dica: Depois de ajustar a árvore usando o pacote `rpart`, tente usar o pacote `caret` para ajustar a árvore e 
# encontrar o valor de cp por validação cruzada.


plotcp(fitcompleto)

# Validação Cruzada
train.control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
model = train(Salary ~., data = treino, method = "rpart", trControl = train.control)
print(model)

# Arvore Resultante
fitmelhor = prune(fitcompleto, cp =0.05418879)
rpart.plot(fitmelhor)

## Parte 2 