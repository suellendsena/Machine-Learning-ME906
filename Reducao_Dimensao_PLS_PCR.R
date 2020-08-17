library(ISLR)
library(tidyverse)
library(caret)
library(ggfortify)
library(pls)
library(ModelMetrics)


## Redução de Dimensão: PCR e PLS

## Objetivos

# Regressão com métodos de redução de dimensão: componentes principais (PCR) e mínimos quadrados parciais (PLS). 
# Será implementado esses métodos no conjunto de dados Hitters, utilizado também no laboratório anterior, 
# e escolher o melhor modelo para predizer o salário  de jogadores de baseball

## Análise exploratória inicial 

data(Hitters)
dados = Hitters %>% select_if(is.numeric) %>% na.omit()


## Separação dos dados 

set.seed(123456)
idtrain <- createDataPartition(dados$Salary, p = 0.8, list=FALSE)

train <- dados %>% slice(idtrain)
test <- dados %>% slice(-idtrain)


## Parte 1 - Regressão com Componentes Principais

pca.train = prcomp(select(train, -Salary), scale = TRUE)
summary(pca.train)


autoplot(pca.train, data = train, colour = "Salary",
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 2,
         scale = 0)


set.seed(2)
pcr.fit = pcr(Salary ~. , data = train, scale = TRUE, validation = "CV")
summary(pcr.fit)

validationplot(pcr.fit, val.type = "RMSEP", las = 1, legendpos = "topright")

# O número de componentes que minimiza o EQM é 4.

pcr.bestfit <- pcr(Salary ~. , ncomp = 4, data = train, scale = TRUE)
summary(pcr.bestfit)

pcr.bestfit$loadings


# PCR EQM_Tr
eqm_tr_pcr = mse(test$Salary, predict(pcr.bestfit, ncomp = 4))


# PCR EQM_Te
pcr.pred = predict(pcr.bestfit, test, ncomp = 4)
eqm_te_pcr = mse(test$Salary, pcr.pred)



## Parte 2 - Regressão com Mínimos Quadrados Parciais

set.seed(2)
pls.fit = plsr(Salary ~. , data = train, scale = TRUE, validation = "CV")
summary(pls.fit)

validationplot(pls.fit, val.type = "RMSEP", las = 1, legendpos = "topright")

# O número de componentes que minimiza o EQM é 1.

pls.bestfit <- plsr(Salary ~. , ncomp = 1, data = train, scale = TRUE)
summary(pls.bestfit)

pls.bestfit$loadings

# PLS EQM_Tr
eqm_tr_pls = mse(test$Salary, predict(pls.bestfit, ncomp = 1))


# PLS EQM_Te
pls.pred = predict(pls.bestfit, test, ncomp = 1)
eqm_te_pls = mse(test$Salary, pls.pred)



## Parte 3 - Comparação entre os modelos

eqm_tr_pcr
eqm_tr_pls


## Resultados 

# O ajuste por PCR apresentou melhores resultados