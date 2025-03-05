# machine_learning
Ejercicios de aprendizaje de máquina con diferentes técnicas

Esta implementación cuenta con moidelos de regresión y clasificación.

## Optimizadores

- Adam
- SGD
- RMSProp

## Debuggers

- simple
- verbose

## Validators

- kfold (k-fold cross validation)
- loo (leave one out)
- MCE (mean classification error)

## Examples to run

python LogisticRegressionFit.py ../../data/telcoCustomer/telco_balanced.csv  -test=0.3 -d="," -o="Adam" -debugger="simple" -v="kfold" -k=5

python LogisticRegressionFit.py ../../data/telcoCustomer/telco_balanced.csv  -test=0.3 -d="," -o="Adam" -debugger="simple" -v="MCE"

python LogisticRegressionFit.py ../../data/telcoCustomer/telco_balanced.csv  -test=0.3 -d="," -o="Adam" -debugger="simple" -v="loo"

## Parametros

- test: porcentaje de datos para el conjunto de prueba
- d: delimitador de los datos
- o: optimizador (Adam, SGD, RMSProp)
- debugger: depurador (simple, verbose)
- v: validador (kfold, loo, MCE)
- k: numero de folds para el validador kfold

## Notas

- El archivo de datos debe estar balanceado, es decir, las clases deben tener la misma cantidad de datos.
- El archivo de datos debe estar en el formato csv.
- El archivo de datos debe tener un encabezado.