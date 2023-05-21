# Projeto de MO810 - Parte I

João Alberto Moreira Seródio - 218548

## Cálculo de Atributos Sísmicos (regressão)

### Modelos

* Regressão Linear - Lasso - Ridge
* Regressão Linear - Lasso - Ridge polinomial
* SVM
* kNN
* Decision Tree
* Random Forest
* Gradient Boosting Decision Tree
* Ensembles

## Classificação de Fáscies Sísmicas (agrupamento)

### Modelos

* K-means
* Hierarchical Clustering (AglomerativeClustering)
* DBSCAN
* Ensembles

## Utilização dos scripts

Criação do DataFrame (CSV chega a 40 GB e dataframe final 20 GB devido a duplicação de dados nas colunas)
```
python create_dataframe.py
```

Split dos Dados em Treino(0.675), Teste(0.225) e Validação (0.1)
```
python data_split.py
```

Criação dos Mini-Datasets de Treino, que pode utilizar técnica de Bagging ou Batching
```
python training_setup.py data_partition -p <DATAFRAME_PATH> -t [batching ou bagging] -o <OUTPUT_PATH> -n <NUMBER_OF_PARTITION>
```

Criação do Mini-Dataset de Teste
```
python training_setup.py data_partition -p <DATAFRAME_PATH> -e
```

Criação do arquivo com as configurações dos modelos a serem treinados
```
python training_setup.py exploration_space -p regression_models -m pre-defined
python training_setup.py exploration_space -p clustering_models -m pre-defined

```

Teste das configurações geradas
```
python training_run.py regression -i -m regression_models -s
python training_run.py clustering -i -m clustering_models -s
```
