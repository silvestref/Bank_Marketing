#------------------------------------------------------------------------------------------------------------
#                                               INTRODUCCIÓN
#------------------------------------------------------------------------------------------------------------

# IDENTIFICACIÓN DEL PROBLEMA

# Uno de los usos mas populares de la ciencia de datos es en el sector del marketing, puesto que es una 
# herramienta muy poderosa que ayuda a las empresas a predecir de cierta forma el resultado de una campaña de
# marketing en base a experiencias pasadas, y que factores seran fundamentales para su exito o fracaso. A la vez
# que tambien ayuda a conocer los perfiles de las personas que tienen mas probabilidad de convertirse en futuros
# clientes con el fin de desarrollar estrategias personalizadas que puedan captar de forma mas efectiva su
# interes. Conocer de antemano o a posteriori esta informacion es de vital importancia ya que ayuda en gran
# medida a que la empresa pueda conocer mas acerca del publico al que se tiene que enfocar y que en el futuro
# se puedan desarrollar campañas de marketing que resulten mas efectivas y eficientes.
# Entonces, se identifica que la problematica a tratar es el entender los factores que influyen a que una 
# persona se suscriba o no a un servicio de deposito a plazo fijo ofrecido por un determinado banco y predecir
# dado una serie de caracteristicas que personas se suscribiran o no a dicho servicio. Para ello, se requiere
# analizar la ultima campaña de marketing ejecutada por el banco y algunas caracteristicas de sus clientes,
# para identificar patrones que nos puedan ayudar a comprender y encontrar soluciones para que
# el banco pueda desarrollar estrategias efectivas que les ayuden a captar el interes de las personas en 
# suscribirse a sus servicios de deposito a plazo fijo, y en base a esto, construir un modelo predictivo que
# permita predecir que personas tomaran este servicio o no.


# ¿QUE ES UN DEPOSITO A PLAZO FIJO?

# Es una inversion que consiste en el deposito de una cantidad determinada de dinero a una institucion
# financiera por un periodo de tiempo, en donde el cliente no puede retirar el dinero depositado hasta que este
# periodo de tiempo haya finalizado. La ventaja de este tipo de deposito es que permite ahorrar dinero ganando
# intereses, por lo cual, muchas personas lo ven como una forma efectiva de generar ingresos pasivos.


# OBJETIVOS

# * Realizar un analisis de datos para encontrar y entender los factores que influyen a que una persona se suscriba
# a un servicio de deposito a plazo fijo.

# * Construir un modelo de aprendizaje automÃ¡tico para la predicción de futuros clientes.


#------------------------------------------------------------------------------------------------------------
#                                   IMPORTACIÓN DE LIBRERIAS Y CARGA DE DATOS
#------------------------------------------------------------------------------------------------------------

# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# El conjunto de datos con el que vamos a tratar almacena características de 11162 personas a los que un banco
# contacto para ofrecerles el servicio de deposito a plazo fijo, e indica si estos al final decidieron adquirir
# dicho servicio o no.
data = pd.read_csv("Bank_Marketing.csv")


#------------------------------------------------------------------------------------------------------------
#                                          EXPLORACIÓN DE LOS DATOS
#------------------------------------------------------------------------------------------------------------

data.head()

data.shape

data.describe()

# Podemos extraer algunos insights simples como que el promedio de edad de los clientes de la empresa rondan
# en los 41 años. Tambien que el saldo promedio que tienen en su cuenta es de 1528, pero si observamos la
# desviacion estandar de los datos de esta variable, observamos que tiene un valor 3225, el cual es considerablemente
# alto, por lo que podemos decir que el saldo de los clientes esta muy distribuido en nuestro conjunto de datos,
# presentando una alta variacion.
# Por ultimo, podemos observar que la variable pdays (numero de dias despues del ultimo contacto en la campaña
# anterior del banco) tiene un valor minimo de -1, lo cual al momento de la interpretabilidad en el analisis de
# datos puede resultar algo confuso, es por ello que en la seccion del preprocesamiento de datos se procedera
# a reemplazar este valor por un 0.

#------------------------------------------
#  ELIMINACIÓN Y CODIFICACIÓN DE VARIABLES
#------------------------------------------

# Hay que tener en cuenta algo de suma importancia en nuestros datos, y es que la variable "duration" hace
# referencia al tiempo de duracion en segundos del ultimo contacto que se realizo con la persona antes que
# decidiera adquirir o no un deposito a plazo fijo, y como naturalmente este valor no se conoce hasta despues
# de haber realizado la llamada que es cuando ya se sabe la decision de la persona, se procedera a eliminar
# al momento de construir nuestro modelo predictivo, puesto que estaria otorgando informacion que de por si
# no se conoce de antemano

data.info()

# Observamos que aparentemente todas nuestras variables de entrada parecen tener cierta relacion con la decision
# de una persona para adquirir o no el servicio de deposito a plazo fijo, por lo que se decide por el momento no
# eliminar ninguna de estas de forma injustificada, esta decision puede cambiar mas adelante con algunas tecnicas
# como la matriz de correlacion.

# Tambien observamos que todas las variables de nuestro conjunto de datos estan correctamente codificadas, por
# lo tanto, no se requiere realizar conversion alguna.


#------------------------------------------------------------------------------------------------------------
#                                           PREPROCESAMIENTO DE DATOS
#------------------------------------------------------------------------------------------------------------

# Como habiamos explicado en la seccion anterior, procederemos a reemplazar los valores iguales a -1 por 0 en
# la variable pdays
for i in range(0,data.shape[0]):
    if data["pdays"][i] == -1:
        data["pdays"][i] = 0
 
# Entonces, si ahora observamos el valor minimo de la variable pdays obtendremos un 0 como resultado
data["pdays"].min()
    
#----------------------------
# IDENTIFICACIÓN DE OUTLIERS
#----------------------------

# Diagramas de caja
fig, ax = plt.subplots(2, 2, figsize=(14,7))
sns.boxplot(ax=ax[0][0], data= data[["age", "day", "campaign", "previous"]], palette="Set3")
sns.boxplot(ax=ax[0][1], data= data[["balance"]], palette="Pastel1")
sns.boxplot(ax=ax[1][0], data= data[["duration"]], palette="Pastel1")
sns.boxplot(ax=ax[1][1], data= data[["pdays"]], palette="Pastel1")
plt.show()

# Con el diagrama de cajas observamos que tenemos presencia de outliers en todas nuestras variables numericas
# excepto en la variables "day".

# A continuacion, visualizaremos el porcentaje de outliers con respecto al total en cada una de nuestras 
# variables para poder considerar si debemos tomar la decision de eliminar alguna de estas variables por su
# alta presencia de valores atipicos

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

( (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)) ).sum() / data.shape[0] * 100

# Los resutados nos arrojan que la variable "pdays" tiene un 24% de presencia de outliers respecto al total de
# filas, lo cual siguiendo la buena practica de eliminar aquellas variables que superen un umbral del 15% de
# valores atipicos, procederemos a eliminar esta variable ya que puede inferir de forma negativa en el analisis
# y la prediccion del futuro modelo de clasificacion que se construira. Otro dato a recalcar, es que esta
# variable es la misma que presentaba valores iguales a -1, los cuales reemplazamos con 0, donde quiza los
# valores etiquetados como -1 se debieron a una corrupcion en los datos, con lo cual tenemos un motivo mas para
# eliminar esta variable.

data = data.drop(["pdays"], axis=1)

# -- AGE --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data[["age"]], palette="Set3")
sns.distplot(data["age"], ax=ax[1])

# De los siguientes graficos podemos observar que los datos que son catalogados como atipicos segun el rango
# intercuartilico son personas que superan los 75 años de edad sin llegar a pasar los 95 años. Este rango de
# edad no es ningun error o corrupcion en los datos, ya que la mayoria de personas con una calidad de vida
# adecuada podrian alcanzar este rango, por lo tanto, tenemos tres opciones para tratarlos.
# Eliminar las filas que contengan estas edades debido a que su presencia es tan solo del 1.5% del total
# Imputarlos haciendo uso de un algoritmo predictivo
# Todos estos metodos resultarian aceptables, pero en este caso optaremos por imputarlos por un valor
# aproximado que refleje la misma conducta que el valor atipico, mas que todo para no perder informacion. De
# igual forma, al momento del entrenamiento y la eleccion del mejor modelo de clasificacion, se comparara el
# rendimiento de un modelo libre de outliers con uno con outliers con el fin de observar si nuestra decision
# fue acertada.

# -- CAMPAIGN --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data[["campaign"]], palette="Set3")
sns.distplot(data["campaign"], ax=ax[1])

# Con respecto a esta variable, observamos que la inmensa mayoria de nuestros datos tienen un valor entre 1 y
# 5, mientras que los datos atipicos adquieren valores superiores a este rango. Evidentemente este es un 
# comportamiento inusual ya que segun nuestros datos, comunmente solo se realizan entre 1 y 5 contactos con el
# cliente antes de que este tome una decision final, por ende, numeros de contactos iguales a 10, 20, 30 e 
# incluso mayores a 40 son demasiado extraños de ver. Por ende, procederemos a imputar estos valores por
# estimaciones que se aproximen a un valor comun.

# -- PREVIOUS --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data[["previous"]], palette="Set3")
sns.distplot(data["previous"], ax=ax[1])

# Al igual que en la variable "campaign", "previous" aparte de tener una difinicion similar (numero de contactos
# con el cliente en la campaña anterior) con "campaing", este tambien presenta un comportamiento equivalente,
# en donde se observa que los valores comunes estan en un rango entre 0 y 3, y que los datos considerados como
# atipicos toman valores superiores a este rango, llegando incluso a ser excesivos. Es por ello que se tomara
# la misma decision de imputarlos al igual que "campaign".

# -- BALANCE --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data[["balance"]], palette="Set3")
sns.distplot(data["balance"], ax=ax[1])

# Un comportamiento similar a las anteriores observamos en esta variable, donde nuevamente tenemos un sesgo
# por la derecha en donde los datos comunes adquieren valores entre -300 y 4000, y los que son atipicos llegan
# a superar facilmente este umbral, aunque resulta mas comun que lo superen en forma positiva que en forma 
# negativa, lo cual podemos deducir que en terminos de valores atipicos, es mas comun encontrar datos anormalmente
# altos que datos anormalmente bajos. Debido a que el procentaje de datos atipicos para esta variable es del
# 9.4%, el cual no es un valor ni muy grande ni muy pequeño, no conviene eliminarlos, es por ello que los
# imputaremos por un nuevo valor aproximado que entre en un rango mas comun.

# -- DURATION --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data[["duration"]], palette="Set3")
sns.distplot(data["duration"], ax=ax[1])

# Esta variable tambien presenta un sesgo notorio por la derecha al igual que las variables anteriores, con la
# diferencia que su distribucion parece ser mas equitativa respecto a las demas, aqui podemos apreciar que 
# los valores comunes estan en un rango entre 0 y 1000 segundos (16 minutos aprox.) y que los que son considerados
# atipicos superan facilmente este rango, llegando incluso a ser superiores a los 3000 segundos (50 minutos).
# Observar que una llamada entre un empleado del banco y un cliente supere los 30 minutos es un comportamiento
# inusual y que no se acostumbra a tener, es por ello que estos datos deben ser tratados, y para este caso
# haremos uso de la imputacion iterativa aplicando bosques aleatorios para reemplazar dichos valores
# por unos que se acerquen a un comportamiento comun de observar.


#----------------------------
# IMPUTACIÓN DE OUTLIERS
#----------------------------

# Crearemos una copia del conjunto de datos original con el fin de mas adelante poder comparar el rendimiento
# de nuestro modelo predictivo en ambos conjuntos

data2 = data.copy()

# El primer paso para realizar la imputacion sera convertir todos los valores atipicos que se hayan detectado
# mediante el rango intercuartilico por NaN, ya que la funcion que utilizaremos para la imputacion trabaja con
# este tipo de datos.

outliers = (data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR))
data2[outliers] = np.nan

# Ahora tenemos que aplicar una codificacion para nuestras variables categoricas, debido a que usaremos bosques
# aleatorios, bastara con aplicar un label encoder

cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "deposit"]

dic = {}

for col in cols:
    dic[col] = LabelEncoder().fit(data2[col])
    data2[col] = dic[col].transform(data2[col])

# El siguiente paso ahora es dividir nuestros datos en conjuntos de entrenamiento y prueba con el fin de evitar
# la fuga de datos.

nom_cols = data2.columns.values

X = data2.iloc[: , :-1].values
y = data2.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)

# Finalmente, procederemos a realizar la imputación

imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=21), random_state=21)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Para visualizar el resultado de nuestra imputacion de forma comoda y grafica sera necesario concatenar todos
# los subconjuntos que hemos creado en uno solo como teniamos inicialmente y revertir la codificacion de
# nuestras variables categoricas

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

data2 = pd.concat([X, y], axis=1)

data2.columns = nom_cols


for col in cols:
    data2[col] = dic[col].inverse_transform(data2[col].astype(int))

# Podemos observar

fig, ax = plt.subplots(2, 2, figsize=(14,7))
sns.boxplot(ax=ax[0][0], data= data2[["age", "day", "campaign", "previous"]], palette="Set3")
sns.boxplot(ax=ax[0][1], data= data2[["balance"]], palette="Pastel1")
sns.boxplot(ax=ax[1][0], data= data2[["duration"]], palette="Pastel1")
plt.show()

Q1 = data2.quantile(0.25)
Q3 = data2.quantile(0.75)
IQR = Q3 - Q1

( (data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR)) ).sum() / data2.shape[0] * 100


