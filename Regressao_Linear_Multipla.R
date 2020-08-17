library(ISLR)
library(tidyverse)
library(ggcorrplot)
library(caret)
library(purrr)
library(ModelMetrics)
library(ggfortify)
library(leaps)

## Regressão Linear Simples e Múltipla

# Sera utilizado o conjunto de dados Carseats, disponível no pacote ISLR, que contém 
# informações de vendas de cadeirinhas de infantis para automóveis de uma determinada marca em 400 lojas 
# diferentes e 11 variáveis

# Sales: unidades de venda em milhares
# CompPrice: preço do competidor (em dólares)
# Income: renda média da comunidade local em 1000 dólares
# Advertising: orçamento local para anúncios em 1000 dólares
# Population: população regional em milhares
# Price: preço da cadeirinha em dólares
# ShelveLoc: qualidade da localização do produto na prateleira (Bad, Good ou Medium)
# Age: idade média da população local (em anos)
# Education: nível médio de educação da população local (em anos)
# Urban: fator indicando se a loja está em uma área urbana ou não.
# US: fator indicando se a loja é nos EUA ou não.


## Objetivo

# O objetivo dessa atividade é encontrar um bom modelo para predizer a variável Sales, 
# baseando-se nas informações disponíveis, usando modelos de regressão linear (simples ou múltipla).

## Análise exploratória inicial

data(Carseats)
glimpse(Carseats)
summary(Carseats)


## Correlação entre as variáveis

# Selecionar apenas os preditores contínuos
dados = Carseats %>% select_if(is.numeric)
corr = cor(dados)

ggcorrplot(corr, method = "square", type = c("lower"), lab = TRUE, lab_size = 2)

## Divisão dos dados

# Aqui, além dos dados treino e teste, utiliza-se também o banco de validação, onde os dados são testados
# antes da aplicação dos modelos no banco de teste.

set.seed(123456)
idTest = createDataPartition(dados$Sales, p = 0.2, list = FALSE)

test = dados %>% slice(idTest)
train0 = dados %>% slice(-idTest)

idTrain = createDataPartition(train0$Sales, p = 0.75, list = FALSE)

train = train0 %>% slice(idTrain)
valid = train0 %>% slice(-idTrain)

## Regressão Linear Simples

train0 %>% ggplot() +
  geom_point(aes(x = Price, y = Sales), color = "blue", alpha = 0.5) +
  labs(title = "Scatterplot between Price and Sales - Training Data")


# Ajusta um modelo de regressao linear Sales ~ xName nos dados de treinamento e calcula os EQMs nos 
# conjuntos 'train' e 'test''

get.eqm.slr = function(train, xName, test){

  train = train %>% select(Sales, xName)
  
  lm.fit = lm(Sales ~ ., data = train) 

  eqm.tr = mse(train$Sales, predict(lm.fit))
  eqm.te = mse(test$Sales, predict(lm.fit, test))
  
  eqms = data.frame(Variavel = xName, EQM_Tr = eqm.tr, EQM_Te = eqm.te)
  return(eqms)
}

preditores = train %>% select(-Sales) %>% colnames()
eqms = preditores %>% map_dfr(~ get.eqm.slr(train, .x, valid)) %>% arrange(EQM_Te)

knitr::kable(eqms) %>% kableExtra::kable_styling(full_width = FALSE)

# Dentre as variáveis contínuas, a variável que apresenta menor EQM nos dados de validação 
# (e também no de treinamento) é Price.
# Ajuste da regressão linear para o preditor Price nos dados de treinamento (juntamente com a parte 
# que foi usada para validação)

slr.fit = lm(Sales ~ Price, data = train0)
summary(slr.fit)

eqm.best = get.eqm.slr(train0, "Price", test)

eqm.slr.final = data.frame(Algorithm = "Regressão Simples", 
                           EQM_Tr = eqm.best$EQM_Tr, 
                           EQM_Te = eqm.best$EQM_Te)
knitr::kable(eqm.slr.final) %>% kableExtra::kable_styling(full_width = FALSE)


## Análise de resíduos

autoplot(slr.fit)


## Regressão Linear Múltipla
# Método de seleção de variáveis Bestsubset

regfit.full = regsubsets(Sales ~ ., train0, nvmax = 10)
reg.summary = summary(regfit.full)

names(reg.summary)
reg.summary$outmat
reg.summary$which


library(tidyr)

reg.result = data.frame(d = 1:7, R2adj = reg.summary$adjr2, 
                        Cp = reg.summary$cp, BIC = reg.summary$bic)

results = reg.result %>% gather(key = "metric", value = "value", -d) 
results.best = results %>% filter(d == 5)

results %>% 
  ggplot(aes(x = d, y = value, color = metric)) +
  geom_point(show.legend = FALSE) +
  geom_line(show.legend = FALSE) + 
  geom_point(data = results.best, color = "black") + 
  labs(x = "Número de Preditores", y = "Valor") + 
  facet_wrap(~ metric, scales = "free_y") 

# De acordo com os gráficos, o número de preditores que minimiza o BIC e Cp e maximiza o R2adj é 5.
# Portanto, os 5 preditores que serão incluídos no modelo são:

coef(regfit.full, 5)

lr.fit = lm(Sales ~ CompPrice + Income + Advertising + Price + Age, data = train0)
summary(lr.fit)

eqm.tr = mse(train0$Sales, predict(lr.fit))
eqm.te = mse(test$Sales, predict(lr.fit, test))

eqm.lr.final = data.frame(Algorithm = "Regressão Múltipla", EQM_Tr = eqm.tr, EQM_Te = eqm.te)
knitr::kable(eqm.lr.final) %>% kableExtra::kable_styling(full_width = FALSE)

## Resultados

eqms.final = rbind(eqm.slr.final, eqm.lr.final)
knitr::kable(eqms.final) %>% kableExtra::kable_styling(full_width = FALSE)

# A regressão múltipla apresenta um desempenho melhor do que a regressão linear simples. 
# Portanto, nosso modelo preditivo final (para essa atividade) é a regressão linear múltipla 
# com os preditores CompPrice, Income, Advertising, Price e Age.
