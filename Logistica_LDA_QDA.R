library(tidyverse)
library(caret)
library(ggplot2)
library(GGally)
library(MASS)
library(kableExtra)
library(nnet)


## Classificadores Binários e Multiclasses: Regressão Logística, LDA e QDA

## Objetivo

# Será implementado os algoritmos de classificação Análise Discrimintante Linear (LDA) e 
# Análise Discriminante Quadrática (QDA) em dois conjuntos de dados:  
# PIMA Indians Diabetes Database, cujo objetivo é diagnosticar se essas pacientes têm ou não diabetes, ou seja,
# uma classificação binária, baseando-se em algumas medidas de diagnóstico. 
# O segundo conjunto de dados é o `iris`, cujo objetivo é classificar plantas em umas das três espécies 
# (setosa, versicolor e virgínica), ou seja, classificação multiclasse, baseando-se em medidas de algumas 
# partes da planta. Para esse segundo conjunto de dados, vocês irão ajustar também um modelo de Regressão 
# Logística Politômica.


## Parte 1

# Nessa parte trabalhar PIMA Indians Diabetes Database. 


bd = read.csv("diabetes.csv")
bd$Outcome = as.factor(bd$Outcome)
aux = createDataPartition(bd$Outcome, p = 0.2, list = FALSE)

bdtrain = bd[aux,]
bdtest = bd[-aux,]
rm(aux)



## Ajustar algoritmo LDA 

ldafit = lda(Outcome ~ ., bdtrain) 
ldapred = predict(ldafit, bdtest)$class

cmlda = confusionMatrix(ldapred, bdtest$Outcome)
cmlda$table

tibble(Acurácia = cmlda$overall[[1]],
       Sensitividade = cmlda$byClass[[1]],
       Especificidade = cmlda$byClass[[2]]) %>% kable() %>% kable_styling()


bdtrain = bdtrain[, c(2, 8, 9)]
bdteste = bdtest[, c(2, 8, 9)]

ldafit = lda(Outcome ~ ., bdtrain) 

grid = expand.grid(Glucose = 0:200, Age = 20:90)
predgrid = predict(ldafit, grid)

grid = cbind(grid, predgrid)
aux = c("Glucose", "Age", "Outcome")
colnames(grid) = aux

ggplot(data = bd, aes(x = Glucose, y = Age, fill = Outcome)) + 
  geom_point(data = grid, aes(x = Glucose, y = Age, colour = Outcome), size = 0.1, alpha = 0.50) +
  geom_point(data = bdtrain, shape = 16, aes(colour = Outcome)) +
  geom_point(data = bdtest, shape = 23, aes(fill = Outcome))



## Aplicar o algoritmo QDA

qdafit = qda(Outcome ~ ., bdtrain)
qdapred = predict(qdafit, bdtest)$class
cmqda = confusionMatrix(qdapred, bdtest$Outcome) 
cmqda$table


tibble(Acurácia = cmqda$overall[[1]],
       Sensitividade = cmqda$byClass[[1]],
       Especificidade = cmqda$byClass[[2]]) %>% kable() %>% kable_styling()


qdafit = qda(Outcome ~ ., bdtrain) 

grid = expand.grid(Glucose = 0:200, Age = 20:90)
predgrid = predict(qdafit, grid)

grid = cbind(grid, predgrid)
aux = c("Glucose", "Age", "Outcome")
colnames(grid) = aux

ggplot(data = bd, aes(x = Glucose, y = Age, fill = Outcome)) + 
  geom_point(data = grid, aes(x = Glucose, y = Age, colour = Outcome), size = 0.1, alpha = 0.50) +
  geom_point(data = bdtrain, shape = 16, aes(colour = Outcome)) +
  geom_point(data = bdtest, shape = 23, aes(fill = Outcome))



  
# Entre os algoritmos de LDA e QDA, observa-se que os melhores resultados foram obtidos ao utilizar LDA.


## Parte 2

# O conjunto de dados `iris` está disponível na base do `R`. 

## Análise exploratória inicial

bd = iris
str(bd)


bd %>% ggplot(aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
bd %>% ggplot(aes(x = Petal.Length, y = Petal.Width, color = Species)) + geom_point()



# Separe 10 observações (20%) de cada espécie para o conjunto de teste. 
# O restante das observações será seu conjunto de treinamento. 
# Faça o pré-processamento dos preditores, se necessário.


set.seed(177261)
aux = createDataPartition(bd$Species, p = 0.2, list = FALSE)

bdtrain = bd[aux,]
bdtest = bd[-aux,]
rm(aux)




# Ajuste uma Regressão Logística Politômica (Multinomial). 
# Apresente métricas relevantes para avaliar o desempenho desse classificador. 


model1 = multinom(Species ~ ., bdtrain)

pred = predict(model1, bdtest)

confu = confusionMatrix(pred, bdtest$Species)
confu$table

tibble(Acurácia = confu$overall[[1]],
       Sensitividade = confu$byClass[[1]],
       Especificidade = confu$byClass[[2]]) %>% kable() %>% kable_styling()


## Ajuste uma Análise Discriminante Linear (LDA). 

ldafit = lda(Species ~ ., bdtrain) 
ldapred = predict(ldafit, bdtest)$class

cmlda = confusionMatrix(ldapred, bdtest$Species)
cmlda$table

tibble(Acurácia = cmlda$overall[[1]],
       Sensitividade = cmlda$byClass[[1]],
       Especificidade = cmlda$byClass[[2]]) %>% kable() %>% kable_styling()



## Ajuste uma Análise Discriminante Linear (QDA). 

qdafit = qda(Species ~ ., bdtrain)
qdapred = predict(qdafit, bdtest)$class
cmqda = confusionMatrix(qdapred, bdtest$Species) 
cmqda$table


tibble(Acurácia = cmqda$overall[[1]],
       Sensitividade = cmqda$byClass[[1]],
       Especificidade = cmqda$byClass[[2]]) %>% kable() %>% kable_styling()


## Gráficos que ilustrem a classificação dos dados no conjunto de teste.


aux1 = seq(0, max(bd$Petal.Width), by=0.05)
aux2 = seq(0, max(bd$Petal.Length), by=0.05)

grid = expand.grid(Petal.Width = aux1, Petal.Length = aux2)
fitlda2 = train(Species ~ Petal.Width + Petal.Length, data = bdtrain, method = "lda")
pred1 = predict(fitlda2, grid)
grid = cbind(grid, pred1)

ggplot(grid, aes(x=Petal.Width, y=Petal.Length, color=pred1)) + geom_point(alpha=0.2, shape=16) + geom_point(data = bdtrain, aes(x=Petal.Width, y=Petal.Length, color=Species), alpha=0.5)  + geom_point(data=bdtest, aes(x=Petal.Width, y=Petal.Length, colour=Species)) + geom_point(data=bdtest, aes(x=Petal.Width, y=Petal.Length), colour="black",shape=23)

grid = expand.grid(Petal.Width=aux1, Petal.Length=aux2)
fitqda2 = train(Species ~ Petal.Width + Petal.Length, data=bdtrain, method="qda")
pred2 = predict(fitqda2, grid)
grid = cbind(grid,pred2)

ggplot(grid, aes(x=Petal.Width, y=Petal.Length, color=pred2)) + geom_point(alpha=0.2, shape=16) + geom_point(data=bdtrain, aes(x=Petal.Width, y=Petal.Length, color=Species),alpha=0.5)  + geom_point(data=bdtest, aes(x=Petal.Width, y=Petal.Length, colour=Species)) + geom_point(data=bdtest, aes(x=Petal.Width, y=Petal.Length), colour="black",shape=23)

grid = expand.grid(Petal.Width=aux1, Petal.Length=aux2)
fitlog2 = train(Species ~ Petal.Width + Petal.Length, data=bdtrain, method="glmnet", family="multinomial", tuneGrid = data.frame(lambda=0, alpha=1))
pred3 = predict(fitlog2, grid)
grid = cbind(grid,pred3)

ggplot(grid, aes(x=Petal.Width, y=Petal.Length, color=pred3)) + geom_point(alpha=0.2, shape=16) + geom_point(data=bdtrain, aes(x=Petal.Width, y=Petal.Length, color=Species),alpha=0.5)  + geom_point(data=bdtest, aes(x=Petal.Width, y=Petal.Length, colour=Species)) + geom_point(data=bdtrain, aes(x=Petal.Width, y=Petal.Length), colour="black",shape=23)


## Resultados
  
# Os modelos de de Regressão Logística Politômica e LDA erraram apenas 2 vezes quanto as predições 
# realizadas, enquanto o modelo QDA errou 5.