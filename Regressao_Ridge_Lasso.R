library(dplyr)
library(ISLR)
library(caret)
library(leaps)
library(tidyverse)
library(glmnet)
library(ModelMetrics)


## Seleção de Variáveis e Regularização

## Objetivos

# Seleção de variáveis utilizando EQM, Ridge e Lasso no banco de dados Hitters para predizer o salário de 
# jogadores de Baseball.


## Análise exploratória inicial

glimpse(Hitters)
summary(Hitters$Salary)
dados = Hitters %>% na.omit()

## Separação dos dados

set.seed(123456)
idtrain <- createDataPartition(dados$Salary, p = 0.8, list=FALSE)
train <- dados %>% slice(idtrain)
test <- dados %>% slice(-idtrain)

## Parte 1 - Seleção de Variáveis em Regressão Linear Múltipla
# seleção best subsets 

bestsubset.fit = regsubsets(Salary ~ ., train, nvmax = 19)
bestsubset.summary = summary(bestsubset.fit)
bestsubset.summary$outmat

names(bestsubset.summary)

bestsubset.metrics = data.frame(d = 1:length(bestsubset.summary$adjr2),
                                R2adj = bestsubset.summary$adjr2,
                                Cp = bestsubset.summary$cp,
                                BIC = bestsubset.summary$bic)
results = bestsubset.metrics %>% gather(key = "metric", value = "value", -d)
rowsBest = c(which.max(bestsubset.metrics$R2adj),
             which.min(bestsubset.metrics$Cp) + length(bestsubset.metrics$d),
             which.min(bestsubset.metrics$BIC) + 2*length(bestsubset.metrics$d))
results.best = results %>% slice(rowsBest)
results %>%
  ggplot(aes(x = d, y = value, color = metric)) +
  geom_point(show.legend = FALSE) +
  geom_line(show.legend = FALSE) +
  geom_point(data = results.best, color = "red") +
  labs(x = "Número de Preditores", y = "Valor") +
  facet_wrap(~ metric, scales = "free_y")

plot(bestsubset.fit, scale = "bic")

coef(bestsubset.fit, 6)


## Ajuste da regressão linear múltipla com seleção de variáveis

lr.bestsubset = lm(Salary ~ Walks + CAtBat + CHits + CHmRun + Division + PutOuts, data = train)
## coef(lr.bestsubset)

## Seleção stepwise

step.fit = regsubsets(Salary ~., data = train, nvmax = 20, method = "seqrep")
step.summary = summary(step.fit)
step.summary$outmat

plot(step.fit, scale = "bic", las = 1)

coef(step.fit, which.min(step.summary$bic))



## Parte 2 - Regressão Ridge e Lasso

# O argumento alpha é fundamental para escolher entre métodos diferentes de regularização. 

x = model.matrix(lm(Salary ~. -1, data = train))
y = train$Salary

ridge.fit = glmnet(x = x, y = y, alpha = 0)
plot(ridge.fit, xvar = "lambda", las = 1)

ridge.cv = cv.glmnet(x, y, alpha = 0)
plot(ridge.cv, las = 1)

lambda.ridge = ridge.cv$lambda.min
## lambda.ridge = ridge.cv$lambda.1se

ridge.best = glmnet(x = x, y = y, alpha = 0, lambda = lambda.ridge)

lasso.fit = glmnet(x = x, y = y, alpha = 1)
plot(lasso.fit, xvar = "lambda", las = 1)

lasso.cv = cv.glmnet(x, y, alpha = 1)
plot(lasso.cv, las = 1)

# Escolha do parâmetro lambda
lambda.lasso = lasso.cv$lambda.min

lasso.best = glmnet(x = x, y = y, alpha = 1, lambda = lambda.lasso)


## Parte 3 - Comparação entre os modelos

# Ajuste da regressão linear múltipla
lr.full = lm(Salary ~ ., data = train)

# Matriz de coeficientes por Regressão, Ridge e Lasso
coef_matrix = cbind(Regressao = coef(lr.full),
                    Ridge = coef(ridge.best),
                    Lasso = coef(lasso.best))
colnames(coef_matrix) = c("Regressão", "Ridge", "Lasso")

coef_matrix


## EQM Regressão Linear - Modelo Completo
eqms.lr = data.frame(Algoritmo = "Regressão Linear Completo",
                     EQM_Tr = mse(train$Salary, predict(lr.full)),
                     EQM_Te = mse(test$Salary, predict(lr.full, test)))
## EQM Regressão Linear com Seleção de Variáveis
eqms.bestsubset = data.frame(Algoritmo = "Seleção Bestsubset",
                             EQM_Tr = mse(train$Salary, predict(lr.bestsubset)),
                             EQM_Te = mse(test$Salary, predict(lr.bestsubset, test)))
## EQM Regressão Ridge
xTest <- model.matrix(lm(Salary ~., data = test))
eqms.ridge = data.frame(Algoritmo = "Regressão Ridge",
                        EQM_Tr = mse(train$Salary, predict(ridge.best, x)),
                        EQM_Te = mse(test$Salary, predict(ridge.best, xTest)))
## EQM Lasso
eqms.lasso = data.frame(Algoritmo = "Lasso",
                        EQM_Tr = mse(train$Salary, predict(lasso.best, x)),
                        EQM_Te = mse(test$Salary, predict(lasso.best, xTest)))
eqms = rbind(eqms.lr, eqms.bestsubset, eqms.ridge, eqms.lasso)
library(knitr)
library(kableExtra)
kable(eqms) %>% kable_styling(bootstrap_options = "striped", full_width = FALSE)

## Resultados

# O modelo de regressão linear completo apresentou melhor desempenho