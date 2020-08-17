
library(tidyverse)
library(ModelMetrics)

## Conceitos básicos 

# Atividade para trabalhar com alguns conceitos básicos de aprendizado de máquina como 
# separação dos dados em treinamento e teste, treinamento do modelo, erro de predição.
# Será utilizado a simualação de dados para a aplicação de modelos de regressão linear simples e polinomial.

set.seed(123456) 
n = 100
x = rnorm(n)
sigma2 = 0.25
eps = rnorm(n, sd = sqrt(sigma2))

# Valores de bota0 e beta1 foram fornecidos pelo exercício
beta0 = -1
beta1 = 0.5
y = beta0 + beta1*x + eps

df = tibble(x = x, y = y)

ggplot() +
  geom_point(aes(x = x, y = y), data = df, col = "blue", alpha = 0.4)

## Separação dos dados em treino e teste 
# Recomenda-se que o banco de dados treino represente 80% dos dados

set.seed(123456)

trainIdx = sort(sample(1:nrow(df), size = 0.8*nrow(df)))

train = df %>% slice(trainIdx)
test = df %>% slice(-trainIdx)

df = df %>% mutate(set = if_else(row_number() %in% trainIdx, "train", "test"))

g1 <- ggplot() +
  geom_point(aes(x = x, y = y, colour = set), data = df, alpha = 0.7)

g1

## Modelo de regressão linear simples por mínimos quadrados

lm.fit = lm(y ~ x, data = train)
summary(lm.fit)

# Utiliza-se o Erro Quadrado Médio (EQM) para medir o poder de precisão do modelo. Ele deve ser utilizado 
# apenas no banco de treino se o objetivo é escolher qual o melhor modelo. A aplicação do EQM no banco teste 
# é apenas para verificar a qualidade do ajuste aos novos os dados. Não se deve escolher o modelo a partir do
# resultado de EQM nos dados.

# EQM treinamento
pred.tr = predict(lm.fit)
EQM_tr = mse(train$y, pred.tr)

# EQM teste
pred.te = predict(lm.fit, test)
EQM_te = mse(test$y, pred.te)

g1 + 
  geom_smooth(aes(x = x, y = y), data = train, 
              method = "lm", se = FALSE, colour = "dimgrey")

# Modelo de regressão polinomial por Mínimos Quadrados

lm2.fit = lm(y ~ x + I(x^2), data = train)
summary(lm2.fit)

EQM2_tr = mse(train$y, predict(lm2.fit))
EQM2_te = mse(test$y, predict(lm2.fit, test))

## Simulando dados novamente aumentando a variabilidade do ruído

set.seed(123456) 
n = 100
x = rnorm(n)
sigma2 = 1
eps = rnorm(n, sd = sqrt(sigma2))
beta0 = -1
beta1 = 0.5
y = beta0 + beta1*x + eps
df = tibble(x = x, y = y)

# Separar em treinamento e teste
set.seed(123456)
trainIdx = sort(sample(1:nrow(df), size = 0.8*nrow(df)))

train = df %>% slice(trainIdx)
test = df %>% slice(-trainIdx)

df = df %>% mutate(set = if_else(row_number() %in% trainIdx, "train", "test"))

# Ajuste da regressão linear
lm.fit = lm(y ~ x, data = train)
summary(lm.fit)

ggplot() +
  geom_point(aes(x = x, y = y, colour = set), data = df, alpha = 0.7) +
  geom_smooth(aes(x = x, y = y), data = train, 
              method = "lm", se=FALSE, colour = "dimgrey")

# EQM treinamento
EQM_tr_k = mse(train$y, predict(lm.fit))

# EQM teste
EQM_te_k = mse(test$y, predict(lm.fit, test))

## Simulando dados com mais observações e menos variabilidade 

set.seed(123456) 
n = 1000
x = rnorm(n)
sigma2 = 0.25
eps = rnorm(n, sd = sqrt(sigma2))
beta0 = -1
beta1 = 0.5
y = beta0 + beta1*x + eps
df = tibble(x = x, y = y)

# Separar em treinamento e teste
set.seed(123456)
trainIdx = sort(sample(1:nrow(df), size = 0.8*nrow(df)))

train = df %>% slice(trainIdx)
test = df %>% slice(-trainIdx)

df = df %>% mutate(set = if_else(row_number() %in% trainIdx, "train", "test"))

# Ajuste da regressão linear
lm.fit = lm(y ~ x, data = train)
summary(lm.fit)

ggplot() +
  geom_point(aes(x = x, y = y, colour = set), data = df, alpha = 0.7) +
  geom_smooth(aes(x = x, y = y), data = train, 
              method = "lm", se=FALSE, colour = "dimgrey")

# EQM treinamento
EQM_tr_l = mse(train$y, predict(lm.fit))

# EQM teste
EQM_te_l = mse(test$y, predict(lm.fit, test))

## Resultados
# Ao comparar os resultados obtidos através do valor de EQM, nota-se que quanto menor a variabilidade nos dados,
# melhor o modelo se ajusta aos dados, independente do valor de n
