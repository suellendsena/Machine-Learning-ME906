library(ISLR)
library(tidyverse)
library(caret)
library(ggcorrplot)
library(ModelMetrics)
library(purrr)
library(FNN)
library(tidyr)

## Regressão KNN

# Será implementado o método KNN no conjunto de dados Hitters para estimar o salário de
# jogadores de baseball e comparar o resultado com o que foi obtido nos últimos dois laboratórios.


## Análise exploratória inicial

data(Hitters)

dados = Hitters %>% select_if(is.numeric) %>% na.omit()

## Separação dos Dados

set.seed(123456)
idtrain <- createDataPartition(dados$Salary, p = 0.8, list=FALSE)
train <- dados %>% slice(idtrain)
test <- dados %>% slice(-idtrain)

# O número de vizinhos k ótimo será estimado usando o conjunto de validação (Parte 1) ou
# validação cruzada (Parte 2).


## Parte 1 - Regressão KNN com um preditor

matcor = cor(train)
ggcorrplot(matcor, type = "upper", lab = TRUE, lab_size = 2)

lr.fit = lm(Salary ~ CRBI, data = train)
summary(lr.fit)$coef

plot.lr = ggplot(aes(x = CRBI, y = Salary), data = train) +
  geom_point(col = "blue", alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  geom_point(data = test, col = "red", alpha = 0.5)
plot.lr


mse.lr.tr = mse(train$Salary, predict(lr.fit))
mse.lr.te = mse(test$Salary, predict(lr.fit, test))
mse.lr = data.frame(Algorithm = "Linear Regression", MSE_Tr = mse.lr.tr, MSE_Te = mse.lr.te)

set.seed(22222)
idtrainv <- createDataPartition(train$Salary, p = 0.8, list=FALSE)
trainv <- dados %>% slice(idtrainv)
valid <- dados %>% slice(-idtrainv)


Xtrainv = trainv["CRBI"]
ytrainv = trainv$Salary

ks <- seq(1, 170, by = 5)
knn.tr <- ks %>% map(~ knn.reg(train = Xtrainv, test = Xtrainv, y = ytrainv, .x)$pred)
knn.mse.tr <- knn.tr %>% map_dbl(~ mse(ytrainv, .x))

Xvalid = valid["CRBI"]
yvalid = valid$Salary
knn.val <- ks %>% map(~ knn.reg(train = Xtrainv, test = Xvalid, y = ytrainv, .x)$pred)
knn.mse.val <- knn.val %>% map_dbl(~ mse(yvalid, .x))

mses = tibble(k = rep(ks, 2),
              mse = c(knn.mse.tr, knn.mse.val),
              set = rep(c("Train", "Validation"), each = length(ks)))
best.k = ks[which.min(knn.mse.val)]

## Gráfico dos EQMs nos conjuntos de treinamento e validação.


mses %>% ggplot(aes(x = k, y = mse, colour = set)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = mse.lr.te, lty = 2) +
  labs(title = "MSE nos conjuntos de Treinamento e Validação",
       y = "MSE", colour = "") +
  ## scale_x_continuous(breaks = ks) +
  theme(legend.position = "bottom")


k.grid <- c(1, 16, 80, 170)
preds <- k.grid %>% map(~ knn.reg(train = Xtrainv, test = Xvalid, y = ytrainv, .x)$pred) %>%
  as.data.frame(col.names = paste0("k", k.grid)) %>%
  mutate(CRBI = valid$CRBI) %>%
  gather("K", "pred", -CRBI) %>%
  mutate(K = factor(K, levels = c("k1", "k16", "k80", "k170")))
labels.k <- c(k1 = "k = 1", k16 = "k = 16", k80 = "k = 80", k170 = "k = 170")
plot.lr +
  geom_step(aes(x = CRBI, y = pred), data = preds,
            stat = "identity", col = "orange") +
  facet_wrap(. ~ K, nrow = 2, labeller = labeller(K = labels.k))

Xtrain = train["CRBI"]
ytrain = train$Salary
Xtest = test["CRBI"]
ytest = test$Salary

# EQM nos dados de treinamento
knn.tr = knn.reg(train = Xtrain, test = Xtrain, y = ytrain, k = best.k)$pred
knn.mse.tr.bestk <- mse(ytrain, knn.tr)

# EQM nos dados de teste
knn.te = knn.reg(train = Xtrain, test = Xtest, y = ytrain, k = best.k)$pred
knn.mse.te.bestk <- mse(ytest, knn.te)

mse.knn = data.frame(Algorithm ="KNN", MSE_Tr = knn.mse.tr.bestk, MSE_Te = knn.mse.te.bestk)
MSE.all = rbind(mse.lr, mse.knn)
knitr::kable(MSE.all) %>% kableExtra::kable_styling(full_width = FALSE)


## Parte 2 - Regressão KNN com todos os preditores

# Padronização dos dados de treinamento e teste

library(caret)
Xtrain = train %>% select(-Salary)
ytrain = train$Salary
Xtest = test %>% select(-Salary)
ytest = test$Salary
preProcPars <- preProcess(Xtrain)
Xtrain <- predict(preProcPars, Xtrain)
Xtest <- predict(preProcPars, Xtest)

# EQM nos dados de treinamento
ks <- seq(1, 170, by = 5)
knn.mse.tr <- ks %>%
  map(~ knn.reg(train = Xtrain, test = Xtrain, y = ytrain, .x)$pred) %>%
  map_dbl(~ mse(ytrain, .x))

# EQM nos dados de teste
knn.mse.te <- ks %>%
  map(~ knn.reg(train = Xtrain, test = Xtest, y = ytrain, .x)$pred) %>%
  map_dbl(~ mse(ytest, .x))
summary(Xtrain[, 1:4])

knnmse <- data_frame(k = ks, MSE_Tr = knn.mse.tr, MSE_Te = knn.mse.te) %>%
  round(3)
# Valor ótimo de k
best.k = ks[which.min(knn.mse.te)]

# O valor de k que minimiza o MSE é k = 6.

trctrl <- trainControl(method = "cv", number = 5)
ks <- data.frame(k = seq(1, 100, by = 5))
set.seed(3456)
knn.fit <- train(Salary ~ ., data = train,
                 method = "knn",
                 trControl = trctrl,
                 preProcess = c("center", "scale"),
                 tuneGrid = ks)

plot(knn.fit, pch = 19)
knn.fit$bestTune

# O valor de k que minimiza o RMSE é dado por k = 6.

mse.knn.tr.all = mse(ytrain, predict(knn.fit))
mse.knn.te.all =mse(ytest, predict(knn.fit, test))
mse.knn.all = data.frame(Algorithm ="KNN All",
                         MSE_Tr = mse.knn.tr.all,
                         MSE_Te = mse.knn.te.all)
MSE.all = rbind(MSE.all, mse.knn.all)
knitr::kable(MSE.all) %>% kableExtra::kable_styling(full_width = FALSE)

## Resultados

# O ajuste realizado com KNN e todas as preditoras do banco foi o que melhor se ajustou aos dados 