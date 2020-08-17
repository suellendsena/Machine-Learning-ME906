library(ISLR)
library(caret)
library(purrr)
library(dplyr)
library(ModelMetrics)

## Conjunto de Validação e Validação Cruzada

## Objetivos

# Utilizar métodos para estimar o erro no conjunto de teste de maneira direta
# usando duas técnicas: conjunto de validação ou validação cruzada.

data(Auto)

getEQM <- function(data, grau.p, replica){

  ## Separação dos Dados
  
  idtrain <- createDataPartition(data$mpg, p = 0.8, list=FALSE)
  train <- data %>% slice(idtrain)
  valid <- data %>% slice(-idtrain)
  
  
  ## Ajuste dos polinômios de graus 1 a grau.p
  fit.poly <- 1:grau.p %>% map(~ lm(mpg ~ poly(horsepower, .), data = train))
  pred_train <- 1:grau.p %>% map(~ predict(fit.poly[.][[1]]))
  pred_valid <- 1:grau.p %>% map(~ predict(fit.poly[.][[1]], valid))
  EQM_train <- 1:grau.p %>% map_dbl(~ mse(train$mpg, pred_train[[.]]))
  EQM_valid <- 1:grau.p %>% map_dbl(~ mse(valid$mpg, pred_valid[[.]]))
  eqm <- data.frame(Grau = factor(rep(1:grau.p, 2)),
                    EQM = c(EQM_train, EQM_valid),
                    Conjunto = rep(c("Treino", "Validação"), each = grau.p),
                    Replica = rep(replica, 2*grau.p))
  return(eqm)
}

## Calcula o EQM de treino e validação para polinômios de graus 1 a 10

eqm <- getEQM(data = ISLR::Auto, 10, replica = 1)


## Calcula o EQM de treino e validação para polinômios de graus 1 a 10 e repete isso 9 vezes

set.seed(4444)
eqm_rep <- map_dfr(1:9, ~ getEQM(data = ISLR::Auto, 10, replica = .), simplify = FALSE)


## Gráfico dos EQMs de treino e validação como função do grau do polonômio

eqm_rep %>% ggplot(aes(x = Grau, y = EQM)) +
  geom_line(aes(group = Replica, color = factor(Replica)), show.legend = FALSE) +
  labs(title="EQM - Regressão Polinomial", x = "Grau do Polinômio") +
  facet_wrap(~ Conjunto)

getEQMcv <- function(data, grau.p, kfold, replica){
  
  ## Separação dos Dados em k-folds
  
  idfold = createFolds(data$mpg, k=kfold)
  EQM_train = EQM_valid = matrix(NA, nrow = grau.p, ncol = kfold)
  
  for(k in 1:kfold){
    train <- data %>% slice(-idfold[[k]])
    valid <- data %>% slice(idfold[[k]])
    ## Ajuste dos polinômios de graus 1 a grau.p
    fit.poly <- 1:grau.p %>% map(~ lm(mpg ~ poly(horsepower, .), data = train))
    pred_train <- 1:grau.p %>% map(~ predict(fit.poly[.][[1]]))
    pred_valid <- 1:grau.p %>% map(~ predict(fit.poly[.][[1]], valid))
    EQM_train[, k] <- 1:grau.p %>% map_dbl(~ mse(train$mpg, pred_train[[.]]))
    EQM_valid[, k] <- 1:grau.p %>% map_dbl(~ mse(valid$mpg, pred_valid[[.]]))
  }
  
  eqm <- data.frame(Grau = factor(rep(1:grau.p, 2)),
                    EQM = c(rowMeans(EQM_train), rowMeans(EQM_valid)),
                    Conjunto = rep(c("Treino", "Validação Cruzada"), each = grau.p),
                    Replica = rep(replica, 2*grau.p))
  return(eqm)
}

## Calcula o EQM de treino e validação cruzada para polinômios de graus 1 a 10

eqm_cv <- getEQMcv(data = ISLR::Auto, 10, kfold = 5, replica = 1)


## Calcula o EQM de treino e validação cruzada para polinômios de graus 1 a 10 e repete isso 9 vezes

set.seed(4444)
eqm_cv_rep <- map_dfr(1:9, ~ getEQMcv(data = ISLR::Auto, 10, kfold = 5, replica = .),
                      simplify = FALSE)


## Gráfico dos EQMs de treino e validação cruzada como função do grau do polinômio

eqm_cv_rep %>% ggplot(aes(x = Grau, y = EQM)) +
  geom_line(aes(group = Replica, color = factor(Replica)), show.legend = FALSE) +
  labs(title="EQM por Validação Cruzada - Regressão Polinomial", x = "Grau do Polinômio") +
  facet_wrap(~ Conjunto)


## Resultados 

# As estimativas do EQM no conjunto de teste obtidas pelo conjunto de validação variam bastante e 
# as conclusões sobre o grau ótimo do polinômio também podem ser bem diferentes. 
# Por outro lado, as estimativas por validação cruzada apresentam uma variabilidade bem menor e, 
# portanto, as conclusões mais consistentes.