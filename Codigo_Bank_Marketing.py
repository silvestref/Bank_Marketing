#------------------------------------------------------------------------------------------------------------
#                                               INTRODUCCI�N
#------------------------------------------------------------------------------------------------------------

# IDENTIFICACI�N DEL PROBLEMA

# Uno de los usos m�s populares de la ciencia de datos es en el sector del marketing, puesto que es una
# herramienta muy poderosa que ayuda a las empresas a predecir de cierta forma el resultado de una campa�a de
# marketing en base a experiencias pasadas, y que factores ser�n fundamentales para su �xito o fracaso. A la
# vez que tambi�n ayuda a conocer los perfiles de las personas que tienen m�s probabilidad de convertirse en
# futuros clientes con el fin de desarrollar estrategias personalizadas que puedan captar de forma m�s efectiva
# su inter�s. Conocer de antemano o a posteriori esta informaci�n es de vital importancia ya que ayuda en gran
# medida a que la empresa pueda conocer m�s acerca del p�blico al que se tiene que enfocar, y que en el futuro
# se puedan desarrollar campa�as de marketing que resulten m�s efectivas y eficientes. Entonces, se identifica
# que la problem�tica a tratar es el entender los factores que influyen a que una persona solicite o no un
# dep�sito a plazo fijo ofrecido por un determinado banco y predecir dado una serie de caracter�sticas, que
# personas solicitar�n o no dicho servicio. Para ello, se requiere analizar la �ltima campa�a de marketing
# ejecutada por el banco y algunas caracter�sticas de sus clientes, con el fin de identificar patrones que nos
# puedan ayudar a comprender y encontrar soluciones para que el banco pueda desarrollar estrategias efectivas
# que les ayuden a captar el inter�s de las personas en solicitar este tipo de dep�sito, y en base a esto,
# construir un modelo predictivo que permita predecir que personas tomaran este servicio o no.


# �QU� ES UN DEP�SITO A PLAZO FIJO?

# Es una inversi�n que consiste en el dep�sito de una cantidad determinada de dinero a una instituci�n
# financiera por un periodo de tiempo, en donde el cliente no puede retirar el dinero depositado hasta que
# este periodo de tiempo haya finalizado. La ventaja de este tipo de dep�sito es que permite ahorrar dinero
# ganando intereses, por lo cual, muchas personas lo ven como una forma efectiva de generar ingresos pasivos.


# OBJETIVOS

# * Realizar an�lisis de datos para encontrar y entender los factores que influyen a que una persona solicite
#   o no un dep�sito a plazo fijo.
# * Construir un modelo de aprendizaje autom�tico con CatBoost para la predicci�n de solicitantes de un dep�sito
#   a plazo fijo.
# * Implementar correctamente cada uno de los pasos de la metodolog�a de ciencia de datos en la elaboraci�n de
#   este proyecto


#------------------------------------------------------------------------------------------------------------
#                                   IMPORTACI�N DE LIBRER�AS Y CARGA DE DATOS
#------------------------------------------------------------------------------------------------------------

# Librer�as
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
from matplotlib.ticker import FormatStrFormatter
import association_metrics as am
from collections import Counter
from catboost import CatBoostClassifier, Pool
import optuna  
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# El conjunto de datos con el que vamos a tratar almacena caracter�sticas de 11162 personas a los que un banco
# contacto para ofrecerles el servicio de deposito a plazo fijo, e indica si estos al final decidieron adquirir
# dicho servicio o no.
data = pd.read_csv("Bank_Marketing.csv")


#------------------------------------------------------------------------------------------------------------
#                                          EXPLORACI�N DE LOS DATOS
#------------------------------------------------------------------------------------------------------------

data.head()

data.shape

data.describe()

# Podemos extraer algunos insights simples de esta tabla, como que el promedio de edad de los clientes de la
# empresa ronda en los 41 a�os. Tambi�n que el saldo promedio que tienen en su cuenta es de 1528, pero si
# observamos la desviaci�n est�ndar de los datos de esta variable, observamos que tiene un valor 3225, el cual
# es considerablemente alto, por lo que podemos decir que el saldo de los clientes est� muy distribuido en
# nuestro conjunto de datos, presentando una alta variaci�n. Por �ltimo, podemos observar que la variable pdays
# (n�mero de d�as despu�s del �ltimo contacto en la campa�a anterior del banco) tiene un valor m�nimo de -1,
# lo cual al momento de la interpretabilidad en el an�lisis de datos puede resultar algo confuso, es por ello
# que en la secci�n del preprocesamiento de datos se proceder� a reemplazar este valor por un 0.

#------------------------------------------
#  ELIMINACI�N Y CAMBIO DE TIPO DE VARIABLES
#------------------------------------------

# Hay que tener en cuenta algo de suma importancia en nuestros datos, y es que la variable "duration" hace
# referencia al tiempo de duraci�n en segundos del �ltimo contacto que se realiz� con la persona antes que
# decidiera solicitar o no un dep�sito a plazo fijo, y como naturalmente este valor no se conoce hasta despu�s
# de haber realizado la llamada que es cuando ya se sabe la decisi�n de la persona, se proceder� a eliminar al
# momento de construir nuestro modelo predictivo, puesto que estar�a otorgando informaci�n que de por si no se
# conoce de antemano.

data.info()

# Observamos que aparentemente todas nuestras variables de entrada parecen tener cierta relaci�n con la decisi�n
# de una persona en solicitar o no un dep�sito a plazo fijo, por lo que se decide por el momento no eliminar
# ninguna de estas variables de forma injustificada.

# Tambi�n observamos que todas las variables de nuestro conjunto de datos est�n correctamente etiquetadas con
# el tipo de dato que les corresponde, por lo tanto, no se requiere realizar conversi�n alguna.


#------------------------------------------------------------------------------------------------------------
#                                           PREPROCESAMIENTO DE DATOS
#------------------------------------------------------------------------------------------------------------

# Como hab�amos explicado en la secci�n anterior, procederemos a reemplazar los valores iguales a -1 por 0 en
# la variable pdays.

for i in range(0,data.shape[0]):
    if data["pdays"].iloc[i] == -1:
        data["pdays"].iloc[i] = 0
 
# Entonces, si ahora observamos el valor m�nimo de la variable pdays obtendremos un 0 como resultado en vez de
# un -1.
data["pdays"].min()
    
#----------------------------
# IDENTIFICACI�N DE OUTLIERS
#----------------------------

# Diagramas de caja
fig, ax = plt.subplots(2, 2, figsize=(14,7))
sns.boxplot(ax=ax[0][0], data= data[["age", "day", "campaign", "previous"]], palette="Set3")
sns.boxplot(ax=ax[0][1], data= data[["balance"]], palette="Pastel1")
sns.boxplot(ax=ax[1][0], data= data[["duration"]], palette="Pastel1")
sns.boxplot(ax=ax[1][1], data= data[["pdays"]], palette="Pastel1")
plt.show()

# Con el diagrama de cajas observamos que tenemos presencia de outliers en todas nuestras variables num�ricas
# excepto en la variable "day".

# A continuaci�n, visualizaremos el porcentaje de outliers con respecto al total en cada una de nuestras
# variables para poder considerar si debemos tomar la decisi�n de eliminar alguna de estas variables por su
# alta presencia de valores at�picos.

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

( (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)) ).sum() / data.shape[0] * 100

# Los resultados nos arrojan que la variable "pdays" tiene un 24% de presencia de outliers respecto al total de
# filas, lo cual siguiendo la buena pr�ctica de eliminar aquellas variables que superen un umbral del 15% de
# valores at�picos, procederemos a eliminar esta variable ya que puede inferir de forma negativa en el an�lisis
# y la predicci�n del futuro modelo de clasificaci�n que se construir�. Otro dato a tomar en cuenta, es que
# esta variable es la misma que presentaba valores iguales a -1, los cuales reemplazamos con 0, donde quiz� los
# valores etiquetados como -1 se debieron a una corrupci�n en los datos, con lo cual tenemos un motivo m�s
# para eliminar esta variable.

data = data.drop(["pdays"], axis=1)

# -- AGE --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["age"]], palette="Set3")
sns.distplot(data["age"], ax=ax[1])
plt.show()

# De los siguientes gr�ficos podemos observar que los datos que son catalogados como at�picos seg�n el rango
# intercuart�lico son personas que superan los 75 a�os de edad sin llegar a pasar los 95 a�os. Este rango de
# edad no es ning�n error o corrupci�n en los datos, ya que la mayor�a de personas con una calidad de vida
# adecuada podr�an alcanzar este rango, por lo tanto, tenemos dos opciones para tratarlos:
    
# * Eliminar las filas que contengan estas edades debido a que su presencia es tan solo del 1.5% del total.
# * Imputarlos haciendo uso de un algoritmo predictivo.

# Todos estos m�todos resultan aceptables, pero en este caso optaremos por imputarlos por un valor aproximado a
# lo "normal" que refleje la misma conducta que el valor at�pico, m�s que todo para no perder informaci�n. De
# igual forma, al momento del entrenamiento y la elecci�n del mejor modelo de clasificaci�n, se comparar� el
# rendimiento de un modelo libre de outliers con uno con outliers con el fin de observar si nuestra decisi�n
# fue acertada.


# -- CAMPAIGN --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["campaign"]], palette="Set3")
sns.distplot(data["campaign"], ax=ax[1])
plt.show()

# Con respecto a esta variable, observamos que la inmensa mayor�a de nuestros datos tienen un valor entre 1 y 5,
# mientras que los datos at�picos adquieren valores superiores a este rango. Evidentemente este es un
# comportamiento inusual ya que, seg�n nuestros datos, com�nmente solo se realizan entre 1 y 5 contactos con el
# cliente antes de que este tome una decisi�n final, por ende, n�meros de contactos iguales a 10, 20, 30 e
# incluso mayores a 40 son demasiado extra�os de ver. Por ende, procederemos a imputar estos valores por
# estimaciones que se aproximen a un valor com�n.

# -- PREVIOUS --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["previous"]], palette="Set3")
sns.distplot(data["previous"], ax=ax[1])
plt.show()

# Al igual que en la variable "campaign", "previous" aparte de tener una definici�n similar (n�mero de contactos
# con el cliente en la campa�a anterior), este tambi�n presenta un comportamiento similar, en donde se observa
# que los valores comunes est�n en un rango entre 0 y 3, y que los datos considerados como at�picos toman
# valores superiores a este rango, llegando incluso a ser excesivos. Es por ello que se tomara la misma
# decisi�n de imputarlos al igual que "campaign".

# -- BALANCE --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["balance"]], palette="Set3")
sns.distplot(data["balance"], ax=ax[1])
plt.show()

# Un comportamiento similar a las anteriores gr�ficas observamos en esta variable, donde nuevamente tenemos un
# sesgo por la derecha en donde los datos comunes adquieren valores entre -300 y 4000, y los que son at�picos
# llegan a superar f�cilmente este umbral, aunque resulta m�s com�n que lo superen en forma positiva que en
# forma negativa, lo cual podemos deducir que, en t�rminos de valores at�picos, es m�s com�n encontrar datos
# anormalmente altos que datos anormalmente bajos. Debido a que el porcentaje de datos at�picos para esta
# variable es del 9.4%, el cual no es un valor ni muy grande ni muy peque�o, no conviene eliminarlos, es por
# ello que los imputaremos por un nuevo valor aproximado que entre en un rango m�s com�n.

# -- DURATION --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["duration"]], palette="Set3")
sns.distplot(data["duration"], ax=ax[1])
plt.show()

# Esta variable tambi�n presenta un sesgo notorio por la derecha al igual que las variables anteriores, con la
# diferencia que su distribuci�n parece ser m�s equitativa respecto a las dem�s, aqu� podemos apreciar que los
# valores comunes est�n en un rango entre 0 y 1000 segundos (16 minutos aprox.) y que los que son considerados
# at�picos superan f�cilmente este rango, llegando incluso a ser superiores a los 3000 segundos (50 minutos).
# Observar que una llamada entre un empleado del banco y un cliente supere los 30 minutos es un comportamiento
# inusual y que no se acostumbra a tener, es por ello que estos datos deben ser tratados, y para este caso
# haremos uso de la imputaci�n iterativa aplicando bosques aleatorios para reemplazar dichos valores por unos
# que se acerquen a un comportamiento com�n de observar.


#----------------------------
# IMPUTACI�N DE OUTLIERS
#----------------------------

# Crearemos una copia del conjunto de datos original con el fin de que mas adelante podamos comparar el
# rendimiento de nuestro modelo predictivo en ambos conjuntos (datos con outliers y sin outliers).

data2 = data.copy()

# El primer paso para realizar la imputaci�n ser� convertir todos los valores at�picos que se hayan detectado
# mediante el rango intercuart�lico por NaN, ya que la funci�n que utilizaremos para la imputaci�n trabaja con
# este tipo de datos.

outliers = (data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR))
data2[outliers] = np.nan

# Ahora tenemos que aplicar una codificaci�n para nuestras variables categ�ricas, debido a que usaremos bosques
# aleatorios como medio de imputaci�n, bastara con aplicar un label encoder.

# Nombres de nuestras variables categ�ricas
cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "deposit"]

# Diccionario para almacenar la codificaci�n realizada en cada variable (�til para despu�s revertir la transformaci�n)
dic = {}

for col in cols:
    dic[col] = LabelEncoder().fit(data2[col])
    data2[col] = dic[col].transform(data2[col])

# El siguiente paso ahora es dividir nuestros datos en conjuntos de entrenamiento y prueba con el fin de evitar
# la fuga de datos.

# Guardamos los nombres de las columnas de nuestro Dataset (�til para despu�s concatenar estos conjuntos en uno solo)
nom_cols = data2.columns.values

X = data2.iloc[: , :-1].values
y = data2.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)

# Finalmente, procederemos a realizar la imputaci�n

imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=21), random_state=21)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Para visualizar el resultado de nuestra imputaci�n de forma c�moda y gr�fica, ser� necesario concatenar todos
# los subconjuntos que hemos creado en uno solo como ten�amos inicialmente y revertir la codificaci�n de
# nuestras variables categ�ricas.

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

data2 = pd.concat([X, y], axis=1)

data2.columns = nom_cols  # Se les introduce los nombres de las columnas con la variable anteriormente creada

# Se invierte la codificaci�n
for col in cols:
    data2[col] = dic[col].inverse_transform(data2[col].astype(int))

# Debido a que las predicciones hechas por los bosques aleatorios se basan en el promedio del resultado de 
# varios arboles de decision, tendremos algunos datos imputados como decimal en variables que son enteras, como
# en el caso de "age", es por ello que redondearemos dichos valores decimales en cada variable que solo 
# contenga valores enteros

for col in ["age", "day", "campaign", "previous", "balance", "duration"]:
    data2[col] = data2[col].round()

# Ahora si podemos graficar para observar el cambio en nuestros datos despu�s de la imputaci�n.

fig, ax = plt.subplots(1, 3, figsize=(16,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data2[["age", "day", "campaign", "previous"]], palette="Set3")
sns.boxplot(ax=ax[1], data= data2[["balance"]], palette="Pastel1")
sns.boxplot(ax=ax[2], data= data2[["duration"]], palette="Pastel1")
plt.show()

# Del grafico podemos observar que todas las variables a excepci�n de "balance" y "duration" est�n libres de
# outliers.

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data2[["balance"]], palette="Set3")
sns.distplot(data2["balance"], ax=ax[1])

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data2[["duration"]], palette="Set3")
sns.distplot(data2["duration"], ax=ax[1])

# Analizando las variables que a�n tienen presencia de valores at�picos, se ve que la varianza en la distribuci�n
# de estos valores ya no es tan extrema como ten�amos inicialmente, si no que ahora se distribuyen en un rango
# menor a 1000 unidades, incluso pudi�ndose acercar a una distribuci�n normal.

Q1 = data2.quantile(0.25)
Q3 = data2.quantile(0.75)
IQR = Q3 - Q1

( (data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR)) ).sum() / data2.shape[0] * 100

# A la vez que tambi�n observamos que estos datos at�picos solo constituyen el 5.6% y 4.1% respectivamente del
# total, lo cual es una cifra moderadamente baja. Entonces podemos tomar dos decisiones, eliminarlos o
# conservarlos como parte de nuestros datos. En esta ocasi�n, elegir� conservarlos ya que pueden contener
# informaci�n �til para el an�lisis y para el modelo de clasificaci�n, adem�s que su presencia es relativamente
# baja con respecto del total y su distancia de los extremos no es tan alarmante ni exagerada.


#------------------------------------
# IDENTIFICACI�N DE VALORES FALTANTES 
#------------------------------------

# Observamos cuantos valores faltantes hay en nuestro conjunto de datos
data2.isnull().sum().sum()

# Debido a que no hay presencia de valores faltantes o nulos, no ser� necesario tomar acciones al respecto



#------------------------------------------------------------------------------------------------------------
#                                      AN�LISIS Y VISUALIZACI�N DE DATOS
#------------------------------------------------------------------------------------------------------------

# En base a las variables que tenemos disponibles, empezaremos la secci�n formulando algunas hip�tesis que seran
# respondidas mediante el proceso de an�lisis de los datos.

# H1: �Es la edad del cliente un factor que propicie la solicitud de un dep�sito a plazo fijo?
# H2: �Qu� tipo de trabajos son m�s propensos a tener clientes que quieran solicitar un dep�sito a plazo fijo?
# H3: �Los clientes casados son menos propensos a solicitar un dep�sito a plazo fijo?
# H4: �El grado de educaci�n alcanzado por el cliente propicia a la solicitud de un dep�sito a plazo fijo?
# H5: �Los clientes con mora crediticia en el banco son menos propensos a solicitar un dep�sito a plazo fijo?
# H6: �Se puede decir que los clientes con mayor dinero en su cuenta bancaria son muy propensos a solicitar un dep�sito a plazo fijo?
# H7: �Los clientes con un pr�stamo para vivienda en el banco son menos propensos a solicitar un dep�sito a plazo fijo?
# H8: �Los clientes con un pr�stamo personal en el banco son menos propensos a solicitar un dep�sito a plazo fijo?
# H9: �El medio de comunicaci�n con el que se contacta al cliente afecta en la solicitud de un dep�sito a plazo fijo?
# H10: �Existen d�as espec�ficos en los que sea m�s probable convencer a un cliente de solicitar un dep�sito a plazo fijo?
# H11: �Existen meses espec�ficos en los que sea m�s probable convencer a un cliente de solicitar un dep�sito a plazo fijo?
# H12: �Se puede decir que a mayor duraci�n en tiempo de contacto con el cliente aumentan las posibilidades de que este acepte solicitar un dep�sito a plazo fijo?
# H13: �Es cierto que mientras m�s contactos se tenga con el cliente mayor ser� la posibilidad de que este termine aceptando solicitar un dep�sito a plazo fijo?
# H14: �El n�mero de contactos realizados en la campa�a anterior afecta en la posibilidad de que los clientes soliciten un dep�sito a plazo fijo?
# H15: �Los clientes que solicitaron un dep�sito a plazo fijo en la campa�a anterior son m�s propensos a solicitar el mismo servicio en la campa�a actual?


#--------------------
# AN�LISIS UNIVARIADO
#--------------------

# Para comenzar, visualizaremos la distribuci�n de los datos respecto a cada uno de los tres conjuntos de
# variables que se han identificado: Variables de informaci�n del cliente - Variables de informaci�n bancaria
# - Variables de campa�a. Esta segmentaci�n nos permitir� realizar un an�lisis m�s ordenado e identificar
# patrones e informaci�n �til para entender nuestros datos.


# VARIABLES DE INFORMACI�N DEL CLIENTE

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.histplot(data=data2, x="age", kde=True, ax=ax[0,0])
ax[0,0].set_title("age")
ax[0,0].set_xlabel("")
sns.countplot(data=data2, x="marital", ax=ax[0,1])
ax[0,1].set_title("marital")
ax[0,1].set_xlabel("")
sns.countplot(data=data2, x="education", ax=ax[1,0])
ax[1,0].set_title("education")
ax[1,0].set_xlabel("")
sns.countplot(data=data2, x="contact", ax=ax[1,1])
ax[1,1].set_title("contact")
ax[1,1].set_xlabel("")
fig.suptitle("Distribuci�n de las variables de informaci�n del cliente", fontsize=16)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
sns.countplot(data=data2, y="job")
ax.set_title("job")
ax.set_ylabel("")
plt.show()

# Observamos que la mayor�a de clientes del banco tienen edades que entran en el rango de los 30 y 40 a�os,
# sin embargo, la diferencia entre el n�mero de clientes que entran en este rango y los que no, no es muy
# grande. Entonces podemos decir que el banco en su gran mayor�a tiene clientes que no sobrepasan la mediana
# edad.

# Tambi�n podemos observar que la mayor�a de estas personas son casadas y que muy pocas son divorciadas. Y que
# el tipo de educaci�n predominante es la secundaria y terciaria.

# Por otra parte, se aprecia que el medio de contacto preferido por los clientes es el celular, y que la mayor�a
# de clientes del banco tienen puestos de gerencia, obrero y t�cnico. Y muy pocos son amas de casa, emprendedores
# o desempleados.


# VARIABLES DE INFORMACI�N BANCARIA

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.histplot(data=data2, x="balance", kde=True, ax=ax[0,0], color="g")
ax[0,0].set_title("balance")
ax[0,0].set_xlabel("")
sns.countplot(data=data2, x="default", ax=ax[0,1])
ax[0,1].set_title("default")
ax[0,1].set_xlabel("")
sns.countplot(data=data2, x="housing", ax=ax[1,0])
ax[1,0].set_title("housing")
ax[1,0].set_xlabel("")
sns.countplot(data=data2, x="loan", ax=ax[1,1])
ax[1,1].set_title("loan")
ax[1,1].set_xlabel("")
fig.suptitle("Distribuci�n de las variables de informaci�n bancaria", fontsize=16)
plt.show()

# Con respecto a la variable "balance" (saldo del cliente en su cuenta bancaria) observamos que existen muchos
# clientes que tienen relativamente poco dinero acumulado en sus cuentas, estos valores se encuentran en un
# rango mayor a 0 y menor a 1000.

# Tambi�n podemos observar que casi no existen clientes morosos en el banco, esta variable se podr�a relacionar
# con "balance", en donde se observa que hay muy pocas personas con saldo negativo en sus cuentas bancarias.

# Por otro lado, tenemos que la cantidad de clientes que han solicitado un pr�stamo para vivienda es muy similar
# a la cantidad de clientes que no solicitaron dicho pr�stamo.

# Por �ltimo, observamos que la gran mayor�a de clientes no han solicitado un pr�stamo personal, y los que si
# lo han hecho, debido a que son minor�a, podr�an relacionarse con la poca presencia de clientes deudores en
# la variable "default" y la poca presencia de clientes con saldo negativo en la variable "balance".


# VARIABLES DE CAMPA�A
fig, ax = plt.subplots(2, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="day", ax=ax[0])
ax[0].set_title("day")
ax[0].set_xlabel("")
ax[0].set_xticklabels(range(1,32))
sns.countplot(data=data2, x="month", ax=ax[1], order=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                                      "nov", "dec"])
ax[1].set_title("month")
ax[1].set_xlabel("")
fig.suptitle('Distribuci�n de las variables de campa�a', fontsize=16)
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.histplot(data=data2, x="duration", kde=True, ax=ax[0,0])
ax[0,0].set_title("duration")
ax[0,0].set_xlabel("")
sns.countplot(data=data2, x="campaign", ax=ax[0,1])
ax[0,1].set_title("campaign")
ax[0,1].set_xlabel("")
ax[0,1].set_xticklabels(range(1,7))
sns.countplot(data=data2, x="previous", ax=ax[1,0])
ax[1,0].set_title("previous")
ax[1,0].set_xlabel("")
ax[1,0].set_xticklabels(range(0,3))
sns.countplot(data=data2, x="poutcome", ax=ax[1,1])
ax[1,1].set_title("poutcome")
ax[1,1].set_xlabel("")
plt.show()

# Observamos que la cantidad de veces con respecto a los d�as en los que se contacta al cliente por �ltima vez
# est�n distribuidos de forma casi equitativa, en donde solo se observan picos muy bajos en los d�as 1, 10,
# 24 y 31 de cada mes, y los picos m�s altos son los que se acercan a principio, quincena o final de cada mes.
# Esto se debe probablemente a que estos d�as son previos al pago que reciben los clientes en su trabajo, por
# ende, se intenta aprovechar el ingreso de dinero que tienen esos d�as para ofrecerles m�todos de inversi�n
# de dinero como el dep�sito a plazo fijo.

# Con respecto al mes en el que se realiz� el �ltimo contacto, se observa que Mayo es un mes con mucha
# importancia para realizar contacto con los clientes, esto podr�a deberse a que los datos que estamos
# analizando provienen de un banco en Portugal, y dicho pa�s celebra en este mes el d�a del trabajo, por lo
# tanto se puede suponer que todos los trabajadores en dicho pa�s reciben un incentivo econ�mico y el banco
# aprovecha esta situaci�n para que el dinero recibido por sus clientes se invierta en la empresa.

# Por otro lado, respecto a la duraci�n del �ltimo contacto en segundos, podemos ver que la gran mayor�a de
# estos tuvo una duraci�n entorno a los 100 y 300 segundos (1.6 y 3.3 minutos respectivamente), lo cual es un
# tiempo justo para saber la decisi�n final del cliente.

# El n�mero de contactos realizados en esta campa�a son en su gran mayor�a 1 o 2. Y el n�mero de contactos
# realizados en la anterior campa�a est� muy inclinado al 0, por lo tanto, se puede deducir que el banco tiene
# nuevos clientes o que la campa�a anterior no fue ejecutada de forma adecuada.

# Por �ltimo, respecto a la variable "poutcome" (resultado de la campa�a anterior) podemos observar que una
# inmensa mayor�a de clientes est�n etiquetados como "unknown" (desconocido), lo cual respalda la suposici�n
# que anteriormente hab�amos hecho respecto a que el banco ten�a nuevo clientes, debido a que esta variable
# guarda relaci�n con "previous".


# Una vez conocida la distribuci�n de las variables con las que vamos a trabajar, procederemos a responder las
# hip�tesis que inicialmente hab�amos planteado, esto lo lograremos mediante un an�lisis bivariado de nuestras
# variables de entrada con nuestra variable de salida.


#-------------------
# ANALISIS BIVARIADO
#-------------------

# VARIABLES DE INFORMACI�N DEL CLIENTE VS "deposit"

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(hspace=0.3)
sns.histplot(data=data2, x="age", kde=True, ax=ax[0,0], hue=data2.deposit, multiple="stack")
ax[0,0].set_title("age")
ax[0,0].set_xlabel("")
sns.countplot(data=data2, x="marital", ax=ax[0,1], hue=data2.deposit, palette="Set2")
ax[0,1].set_title("marital")
ax[0,1].set_xlabel("")
sns.countplot(data=data2, x="education", ax=ax[1,0], hue=data2.deposit, palette="Set2")
ax[1,0].set_title("education")
ax[1,0].set_xlabel("")
sns.countplot(data=data2, x="contact", ax=ax[1,1], hue=data2.deposit, palette="Set2")
ax[1,1].set_title("contact")
ax[1,1].set_xlabel("")
fig.suptitle('Variables de informaci�n del cliente vs deposit', fontsize=16)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
sns.countplot(data=data2, x="job", hue=data2.deposit)
ax.set_title("job")
ax.set_ylabel("")
plt.show()

# En primer lugar, mediante el histograma observamos que la curva de densidad de las edades de los clientes que
# solicitaron y no solicitaron el dep�sito son muy similares, obteniendo en ambos casos los picos m�s altos en
# edades que entran en el rango de los 30 y 40 a�os, y que estos picos se diferencian por relativamente pocas
# cifras de las dem�s edades. Es decir, no existe un patr�n claro que indique de forma significativa que una
# edad en espec�fico es m�s propensa a solicitar un dep�sito a plazo fijo o no.

# Por otro lado, podemos observar que la variable "marital" no presenta relaci�n alguna con la solicitud de un
# dep�sito a plazo fijo, ya que la cantidad de clientes que solicitaron o no el dep�sito se reparten de forma
# equitativa entre los que son solteros, casados y divorciados

# El mismo comportamiento se puede apreciar en la variable "education", donde la cantidad de clientes solicitantes
# y no solicitantes son muy parecidas en todos los grados de educaci�n.

# Con respecto a "contact", podemos identificar que los clientes con un medio de comunicaci�n desconocido por
# el banco son menos propensos a solicitar un dep�sito a plazo fijo, esta informaci�n podr�a no ser tan
# relevante debido a que como el medio de comunicaci�n es desconocido, estos datos podr�an ir a cualquier de
# las dos categor�as restantes, sesgando un poco el resultado del an�lisis.

# Por �ltimo, en la variable "job" podemos observar que los clientes con trabajo "blue-collar" (obrero) son
# menos propensos a solicitar un dep�sito a plazo fijo, probablemente por los pocos ingresos que se obtienen
# de esta labor. Por otra parte, observamos que los estudiantes (student) y los retirados (retired) son
# levemente m�s propensos a solicitar este tipo de dep�sito, posiblemente debido a la cultura financiera que
# existe en la mayor�a de centros educativos y la alta disponibilidad de dinero que se tiene al haberse
# jubilado.

# Resumiendo toda la informaci�n obtenida tenemos que: La edad de los clientes no es un factor muy influyente
# para determinar si estos van a solicitar un dep�sito a plazo fijo o no, adem�s que tanto su estado marital
# como educacional tampoco influyen en esta decisi�n, sin embargo, se observa que los clientes con un medio de
# contacto desconocido por el banco son m�s propensos a no solicitar este tipo de dep�sito, a la vez que los
# que tienen trabajos relacionados con la mano de obra tienen una tendencia a tampoco solicitar este servicio,
# y las personas que son estudiantes o retiradas a menudo aceptan el dep�sito a plazo fijo.

# Respondiendo a las hip�tesis tenemos que:
# H1: La edad del cliente no afecta de forma significativa en la decisi�n de solicitar un dep�sito a plazo fijo.
# H2: Se observo que los estudiantes y las personas retiradas son ligeramente m�s propensas a solicitar un dep�sito a plazo fijo.
# H3: El estado marital del cliente no influye en la decisi�n de solicitar un dep�sito a plazo fijo.
# H4: El grado de educaci�n alcanzado por el cliente no influye de forma significativa en la decisi�n de solicitar un dep�sito a plazo fijo.
# H9: Los clientes con un medio de contacto desconocido por el banco tienen ligeramente m�s probabilidad de no solicitar un dep�sito a plazo fijo.


# VARIABLES DE INFORMACI�N BANCARIA VS "deposit"

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(hspace=0.3)
sns.histplot(data=data2, x="balance", kde=True, ax=ax[0,0], hue=data2.deposit, multiple="stack")
ax[0,0].set_title("balance")
ax[0,0].set_xlabel("")
sns.countplot(data=data2, x="default", ax=ax[0,1], hue=data2.deposit, palette="Set2")
ax[0,1].set_title("default")
ax[0,1].set_xlabel("")
sns.countplot(data=data2, x="housing", ax=ax[1,0], hue=data2.deposit, palette="Set2")
ax[1,0].set_title("housing")
ax[1,0].set_xlabel("")
sns.countplot(data=data2, x="loan", ax=ax[1,1], hue=data2.deposit, palette="Set2")
ax[1,1].set_title("loan")
ax[1,1].set_xlabel("")
fig.suptitle('Variables de informaci�n bancaria vs deposit', fontsize=16)
plt.show()

# Del primer gr�fico, mediante el histograma observamos que las curvas de densidad del saldo de los clientes
# en el banco son muy similares para los que solicitaron y no solicitaron el dep�sito a plazo fijo, siguiendo
# casi una distribuci�n normal, no se observa alg�n patr�n marcado que indique que un rango de saldo en
# espec�fico propicie o no la solicitud de un dep�sito a plazo fijo.

# Con respecto a "default", tampoco se puede observar alg�n patr�n especifico que indique que el ser un cliente
# moroso o no afecte en la decisi�n de solicitar o no un dep�sito a plazo fijo, ya que la distribuci�n de estos
# se reparte de forma equitativa en ambas ocasiones.

# Con "housing" no podemos decir lo mismo, ya que aqu� si se aprecia que los clientes que no solicitaron un
# pr�stamo de vivienda tienen una tendencia a solicitar un dep�sito a plazo fijo, mientras que los que si
# solicitaron un pr�stamo de vivienda tienen una tendencia a no solicitar este tipo de dep�sito. Esto podr�a
# deberse a que como ya tienen una deuda con el banco, ese dinero solicitado no puede destinarse a otros fines
# que no sea la adquisici�n de una propiedad.

# Por �ltimo, en la variable "loan" nuevamente no se observa un patr�n claro que indique una inclinaci�n hacia
# solicitar o no solicitar un dep�sito a plazo fijo si el cliente ha solicitado un pr�stamo personal o no.

# Resumiendo toda la informaci�n obtenida tenemos que: El saldo de los clientes en sus cuentas bancarias no es
# un factor del que se pueda deducir si estos en un futuro solicitaran un dep�sito a plazo fijo o no, lo mismo
# podemos decir con respecto a si este cliente tiene mora crediticia o no, y si este solicito un pr�stamo
# personal o no. Sin embargo, observamos un patr�n claro que indica que los clientes que solicitaron un
# pr�stamo de vivienda son menos propensos a solicitar un dep�sito a plazo fijo, probablemente porque ese
# dinero solicitado ser� destinado a otros fines.

# Respondiendo a las hip�tesis tenemos que:
# H5: El hecho de tener o no tener mora crediticia no influye en la decisi�n de solicitar o no un dep�sito a plazo fijo.
# H6: El dinero que los clientes tengan en su cuenta bancaria no influye en la decisi�n de solicitar o no un dep�sito a plazo fijo.
# H7: Los clientes que solicitaron un pr�stamo de vivienda al banco son menos propensos a solicitar un dep�sito a plazo fijo.
# H8: El hecho de solicitar o no un pr�stamo personal al banco no influye de forma significativa en la decisi�n de solicitar o no un dep�sito a plazo fijo.


# VARIABLES DE CAMPA�A VS "deposit"

fig, ax = plt.subplots(2, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="day", hue=data2.deposit, ax=ax[0])
ax[0].set_title("day")
ax[0].set_xlabel("")
ax[0].set_xticklabels(range(1,32))
sns.countplot(data=data2, x="month", ax=ax[1], order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                                                      'sep', 'oct', "nov", "dec"], hue=data2.deposit)
ax[1].set_title("month")
ax[1].set_xlabel("")
fig.suptitle('Variables de campa�a vs deposit', fontsize=16)
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.histplot(data=data2, x="duration", kde=True, hue=data2.deposit, ax=ax[0,0], multiple="stack")
ax[0,0].set_title("duration")
ax[0,0].set_xlabel("")
sns.countplot(data=data2, x="campaign", hue=data2.deposit, ax=ax[0,1])
ax[0,1].set_title("campaign")
ax[0,1].set_xlabel("")
sns.countplot(data=data2, x="previous", hue=data2.deposit, ax=ax[1,0])
ax[1,0].set_title("previous")
ax[1,0].set_xlabel("")
sns.countplot(data=data2, x="poutcome", hue=data2.deposit, ax=ax[1,1])
ax[1,1].set_title("poutcome")
ax[1,1].set_xlabel("")
plt.show()

# En primer lugar podemos observar que los d�as en donde se tuvo mayor �xito con respecto a la solicitud de un
# dep�sito a plazo fijo son los d�as 1 y 10 de cada mes, el �xito que se tiene en el d�a 1 puede deberse a que
# este es un d�a despu�s del que los clientes reciben su pago mensual por laborar, entonces, al tener una
# cantidad considerable de dinero en sus manos, es m�s f�cil persuadirlos para que lo inviertan en el banco,
# tambi�n podemos ver que este d�a es uno de los que menos contacto se tiene con el cliente, por lo tanto se
# podr�a recomendar para la pr�xima campa�a aprovechar este d�a para persuadir a m�s personas. Con respecto a
# los d�as en los que no se tuvo �xito, podemos observar que estos son el 19, 20, 28 y 29 de cada mes.

# Por otra parte, observamos que los meses en los que se tuvo mayor �xito fueron marzo, septiembre, octubre y
# en menor medida diciembre, mientras que el mes en el que se tuvo un mayor fracaso fue mayo.

# Con respecto a la variable "duration", se puede observar que en un principio los pocos segundos de comunicaci�n
# con el cliente tienen una proporci�n similar de clientes que solicitaron y no solicitaron el dep�sito, y que
# a medida que el tiempo de contacto se vaya prolongando, hay mejores probabilidades de que este termine
# aceptando realizar este tipo de dep�sito. Esta es una conducta normal, ya que cuando una persona est�
# interesada en adquirir alg�n producto o servicio, surgen diversas preguntas acerca de ello, lo cual,
# naturalmente prolonga el tiempo de comunicaci�n con el individuo o entidad que brinda dicho servicio.

# En la variable "campaign" se puede apreciar que no existe un patr�n que indique con certeza que un determinado
# n�mero de contactos favorece a la solicitud de un dep�sito a plazo fijo, aunque observamos que existen m�s
# clientes que aceptaron realizar el dep�sito cuando se realiz� solo 1 contacto con ellos, la diferencia entre
# los que aceptaron o no, no es muy grande para considerarlo relevante.

# Con "previous" no podemos decir lo mismo, ya que se ve que los clientes que no han sido contactados en una
# campa�a anterior para ofrecerles este tipo de dep�sito son menos propensos a aceptar dicho dep�sito en la
# campa�a actual, mientras que aquellos que si han sido contactados anteriormente, tienen una leve 
# a solicitar este tipo de servicio.

# Por �ltimo, respecto a la variable "poutcome", podemos observar que aquellos clientes de los que no se sabe
# si aceptaron o no solicitar un dep�sito a plazo fijo en la campa�a anterior tienen una tendencia a no
# aceptar este tipo de dep�sito en la campa�a actual, cabe mencionar que si su decisi�n fue etiquetada como
# desconocida, podr�a deberse a que son nuevos clientes, ya que esta variable guarda relaci�n con "previous",
# en donde se puede observar que la cantidad de clientes a los que no se les han contactado en la campa�a
# anterior (0) es la misma que los que los que est�n etiquetados como "unknown". Por otro lado, observamos que
# aquellos clientes a los cuales se les pudo persuadir para solicitar este tipo de dep�sito en la campa�a
# anterior, con mucha probabilidad volver�n a aceptar solicitar este servicio en la campa�a actual.

# Resumiendo toda la informaci�n obtenida tenemos que: Los d�as que registraron mayor �xito en la solicitud de
# un dep�sito a plazo fijo fueron los 1 y 10 de cada mes, mientras que los que menos �xito tuvieron fueron los
# d�as 19, 20, 28 y 29. Asimismo los meses de mayor �xito fueron Marzo, Septiembre, Octubre y Diciembre, y el
# de menor �xito Mayo. Tambi�n se observ� que si se tiene una comunicaci�n corta con el cliente, la
# posibilidad que este acepte solicitar este tipo de dep�sito es casi la misma que la de no solicitarlo, y que
# mientras mayor sea el tiempo de contacto, mayor ser� la posibilidad de tener �xito en su persuasi�n. El
# n�mero de contactos que se tiene con el cliente parece no afectar en su decisi�n, sin embargo, variables
# referentes a la campa�a anterior como "previous" y "poutcome" parecen si afectar en esta decisi�n, donde se
# pudo identificar que aquellos clientes a los cuales no se les contact� en la campa�a anterior y cuyo resultado 
# de si aceptaron solicitar el dep�sito a plazo fijo o no es desconocido, tienen una tendencia a no solicitar
# este tipo de dep�sito en la campa�a actual, mientras que aquellos de los que se sabe que si aceptaron
# solicitar este dep�sito en la campa�a anterior, con mucha probabilidad volver�n a solicitarlo en la campa�a
# actual.

# Respondiendo a las hip�tesis tenemos que:
# H10: Los d�as en los que se observ� que es m�s probable convencer a los clientes de solicitar un dep�sito a plazo fijo fueron el 1 y 10 de cada mes.
# H11: Los meses en los que se observ� que es m�s probable convencer a los clientes de solicitar un dep�sito a plazo fijo fueron marzo, septiembre, octubre y diciembre.
# H12: A mayor duraci�n en el tiempo de contacto con el cliente mayores posibilidades hay de que este termine aceptando solicitar un dep�sito a plazo fijo.
# H13: El n�mero de contactos que se tiene con el cliente parece no afectar en su decisi�n de solicitar o no un dep�sito a plazo fijo.
# H14: El n�mero de contactos que se tuvo con el cliente en la campa�a anterior afecta en la posibilidad de solicitar un dep�sito a plazo fijo.
# H15: Aquellos clientes que solicitaron un dep�sito a plazo fijo en la campa�a anterior con mucha probabilidad volver�n a solicitar este tipo de dep�sito en la campa�a actual.


# A lo largo del proceso de an�lisis para responder las hip�tesis que inicialmente hab�amos planteado, nos
# hemos encontrado con algunos comportamientos y patrones particulares de los cuales se puede extraer
# informaci�n relevante para el an�lisis. Es por ello que en esta parte se compararan entre s� algunas de las
# variables m�s relevantes con respecto a la decisi�n de solicitar un dep�sito a plazo fijo con el fin de
# obtener insights que nos ayuden a entender un poco m�s el comportamiento de los clientes.


# "job" vs "housing"

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="job", hue=data2.housing)
ax.set_title("job")
ax.set_xlabel("")
plt.show()

# Del grafico podemos observar que los clientes que trabajan de obrero son los que en su gran mayor�a
# solicitan un pr�stamo de vivienda, es por ello que las personas con este tipo de trabajo son menos propensas
# a solicitar un dep�sito a plazo fijo, ya que como vimos en an�lisis anteriores, el dinero que piden prestado
# al banco va destinado a otros fines que no son los buscados en este an�lisis. Por otra parte, podemos observar
# que las personas que son estudiantes o retirados son menos propensas a solicitar un pr�stamo de vivienda, por
# lo tanto, uniendo los hilos con el an�lisis respecto a la variable "housing", es de esperar que estas personas
# tengan m�s probabilidades de solicitar un dep�sito a plazo fijo puesto que no tienen deudas con el banco, y
# probablemente su cultura financiera o experiencia les hace m�s atractivo el hecho de invertir que de gastar.


# "previous" vs "poutcome"

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="previous", hue=data2.poutcome)
ax.set_title("previous")
ax.set_xlabel("")
plt.show()

# Del siguiente grafico obtenemos un patr�n muy obvio en donde los clientes que no han sido contactados en la
# campa�a anterior, est�n etiquetados como resultado desconocido en si solicitaron o no un dep�sito a plazo
# fijo en la campa�a anterior. Por otra parte, podemos observar que efectivamente el n�mero de contactos que
# se tiene con el cliente no afecta en su decisi�n de solicitar o no este tipo de dep�sito, ya que como se
# puede apreciar, las personas que solicitaron y no solicitaron el dep�sito se distribuyen de forma muy
# equitativa.


# "job" vs "duration"

duration_mean = data2.groupby(["job"], as_index=False)["duration"].mean()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.barplot(data=duration_mean, x="duration", y="job")
ax.set_title("previous")
ax.set_xlabel("")
plt.show()

# Por �ltimo, podemos observar que la media de tiempo de contacto que se tiene con cada uno de los clientes
# pertenecientes a los distintos tipos de trabajo se distribuye de forma muy equitativa, donde la diferencia
# m�xima que se puede apreciar es de 1 minuto. Aunque podemos ver que el tiempo de contacto que se tienen con
# los clientes que se encuentran desempleados es ligeramente mayor al resto, esto podr�a deberse a que la
# situaci�n de estas personas les obliga a tener una fuente de ingresos para poder subsistir, por lo tanto,
# el tiempo de contacto con ellos se ve m�s prolongado al tener m�s inter�s en consultar como es el
# funcionamiento de este tipo de dep�sito y sus beneficios.


# Para terminar con esta secci�n, graficaremos una matriz de correlaci�n para identificar el comportamiento
# conjunto de nuestras variables sobre otras. Como estamos tratando tanto con variables categ�ricas como
# num�ricas, ser� necesario aplicar la correlaci�n de Pearson para las caracter�sticas num�ricas, y la V de
# Cramer para las categ�ricas.


#-----------------------
# CORRELACION DE PEARSON
#-----------------------

data_corr = data2.copy()

data_corr["deposit"] = LabelEncoder().fit_transform(data_corr["deposit"])

plt.figure(figsize=(18, 10))
corr = data_corr[["age","balance","day","duration","campaign","previous","deposit"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)

# De la matriz observamos que las variables num�ricas con mayor correlaci�n hacia nuestra variable dependiente
# son "duration" y "previous". La influencia de estas variables ya lo hab�amos analizado y gracias a esta
# matriz nuestras suposiciones est�n mejor respaldadas. Respecto a "duration", hab�amos llegado a la conclusi�n
# que mientras mayor era el n�mero de segundos en el que se manten�a contacto con el cliente, mayores eran las
# posibilidades de que este terminara aceptando solicitar el dep�sito. Y con respecto a "previous", identificamos
# que los clientes que no hab�an sido contactados en una campa�a anterior, ten�an m�s probabilidad de no
# solicitar el dep�sito en la campa�a actual.


#------------
# V DE CRAMER
#------------

data_corr = data2.copy()

data_corr = data_corr.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
cramersv = am.CramersV(data_corr) 
result = cramersv.fit()

# Con respecto a la asociaci�n entre nuestras variables categ�ricas y nuestra variable dependiente podemos
# observar que aquellas cuyo valor de asociaci�n es mayor que el resto son "housing", "contact", "month" y
# "poutcome", las cuales en an�lisis anteriores hab�amos observado que presentaban ciertos patrones que indicaban
# la inclinaci�n del cliente hacia solicitar o no solicitar un dep�sito a plazo fijo. Donde los clientes que
# solicitaron un pr�stamo de vivienda eran menos propensos a solicitar este tipo de dep�sito, al iguales que
# los clientes de los que no se conoc�a el medio de comunicaci�n por el cual se les contactaba. Con respecto a
# los meses, observamos que hab�a algunos en los que se ten�an resultados muchos m�s positivos y otros en los
# que no hab�a mucho �xito. Por �ltimo, tambi�n pudimos identificar que aquellos clientes que hab�an solicitado
# realizar este tipo de dep�sito en la campa�a anterior con mucha probabilidad volver�an a solicitarlo en la
# campa�a actual, mientras que aquellos que eran nuevos en el banco y no se ten�a un registro acerca de su
# decisi�n, ten�an una tendencia a no solicitar este servicio.

# Gracias al valor de la asociaci�n de Cramer tambi�n podemos obtener algunos otros insights interesantes como:
    
# "month" vs "housing"

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.countplot(data=data2, x="month", order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', "nov", "dec"],
              hue=data2.housing)
plt.show()

# En donde podemos observar que existe un n�mero significativo de clientes que han sido contactados por �ltima
# vez en mayo y que han solicitado un pr�stamo de vivienda. Esto podr�a indicar que los clientes tienen una
# tendencia a solicitar este pr�stamo un mes antes de Mayo, ya que se puede observar c�mo en Abril el n�mero de
# personas con esta solicitud van en aumento, y como pasado el mes de Mayo este n�mero decrece, volviendo a un
# estado est�ndar en el que el n�mero de personas que no solicitaron este tipo de pr�stamos son mayores o iguales
# a las que si lo solicitaron.


# "job" vs "education"

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.countplot(data=data2, x="job", hue=data2.education)
ax.set_title("job")
ax.set_xlabel("")
plt.show()

# Por �ltimo, podemos observar un patr�n completamente normal en donde la mayor�a de personas que tienen cargos
# relacionados con la gerencia, tienen estudios terciarios (universitarios o de instituto). Y que los dem�s
# puestos de trabajo est�n conformados por personas cuyo grado de educaci�n mayormente es secundario, excepto
# en el caso de los obreros, retirados y amas de casa, donde la distribuci�n entre las personas con educaci�n
# secundaria y primaria es casi equitativa.

#------------------------------------------------------------------------------------------------------------
#                                           TRANSFORMACI�N DE DATOS
#------------------------------------------------------------------------------------------------------------

# Empezaremos por eliminar la variable "duration" de la cual anteriormente hab�amos hablado, ya que aporta
# informaci�n de la cual no se dispone en la realidad al momento de predecir si un cliente solicitara o no un
# dep�sito a plazo fijo, ya que la duraci�n de la llamada con el cliente se conoce despu�s de saber la decisi�n
# de este, mas no antes.

data = data.drop(["duration"], axis=1)
data2 = data2.drop(["duration"], axis=1)

#---------------------------
# CODIFICIACI�N DE VARIABLES
#---------------------------

# Como uno de los objetivos de este proyecto es implementar CatBoost para la predicci�n de clientes que solicitaran
# o no un dep�sito a plazo fijo en el futuro, no ser� necesario codificar de forma manual nuestras variables
# categ�ricas, ya que CatBoost internamente realiza este proceso por nosotros, implementando una codificaci�n
# basada en Target Encoder con algunas modificaciones que el algoritmo cree pertinente. Solo ser�a necesario
# aplicar una codificaci�n de etiqueta a nuestra variable dependiente solo si esta es dicot�mica. Sin embargo,
# para demostrar que efectividad tiene el delegarle la codificaci�n a CatBoost y hacerlo de forma manual en la
# precisi�n de nuestro modelo, construiremos dos modelos utilizando ambas t�cnicas y posteriormente evaluaremos
# su rendimiento.


# CON CODIFICACI�N MANUAL
#------------------------

# Puesto que el algoritmo que vamos a utilizar est� basado en �rboles de decisi�n, para evitar el aumento
# exponencial de variables independientes al implementar una codificaci�n One Hot Econding y todos los
# problemas que esto conlleva, podemos utilizar Label Encoder como alternativa, ya que los �rboles de decisi�n
# no se ven perjudicados al tener variables ordinales que originalmente son nominales.

# Codificaci�n de variables en el conjunto con outliers
data_cod = data.copy()

cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "deposit"]

for col in cols:
    data_cod[col] = LabelEncoder().fit_transform(data_cod[col])
    
# Codificaci�n de variables en el conjunto sin outliers

data2_cod = data2.copy()

for col in cols:
    data2_cod[col] = LabelEncoder().fit_transform(data2_cod[col])


# SIN CODIFICACI�N MANUAL
#------------------------

# Codificaci�n de etiqueta a la variable dependiente del conjunto con outliers 
data["deposit"] = LabelEncoder().fit_transform(data["deposit"])

# Codificaci�n de etiqueta a la variable dependiente del conjunto sin outliers 
data2["deposit"] = LabelEncoder().fit_transform(data2["deposit"])


#----------------------------------------------------
# CREACI�N DE CONJUNTOS DE ENTRENAMIENTO Y VALIDACI�N
#----------------------------------------------------

# PARA DATOS CON OUTLIERS Y SIN CODIFICACI�N MANUAL
#--------------------------------------------------
X = data.iloc[: , :-1].values
y = data.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)


# PARA DATOS CON OUTLIERS Y CON CODIFICACI�N MANUAL
#--------------------------------------------------
X_cod = data_cod.iloc[: , :-1].values
y_cod = data_cod.iloc[: , -1].values

X_train_cod, X_test_cod, y_train_cod, y_test_cod = train_test_split(X_cod, y_cod, test_size=0.30,
                                                                    random_state=21, stratify=y)


# PARA DATOS SIN OUTLIERS Y SIN CODIFICACI�N MANUAL
#--------------------------------------------------
X2 = data2.iloc[: , :-1].values
y2 = data2.iloc[: , -1].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.30, random_state=21, stratify=y)


# PARA DATOS SIN OUTLIERS Y CON CODIFICAC�N MANUAL
#--------------------------------------------------
X2_cod = data2_cod.iloc[: , :-1].values
y2_cod = data2_cod.iloc[: , -1].values

X2_train_cod, X2_test_cod, y2_train_cod, y2_test_cod = train_test_split(X2_cod, y2_cod, test_size=0.30,
                                                                        random_state=21, stratify=y)


#--------------------
# REBALANCEO DE DATOS
#--------------------

# Empezaremos comprobando el n�mero de muestras para cada una de las clases que tiene nuestra variable dependiente
# para identificar si tenemos un conjunto de datos desbalanceado.

plt.figure(figsize=(15, 8))
sns.countplot(data=data2, x="deposit", palette=["#66c2a5", "#fc8d62"])
plt.title("Distribuci�n del n�mero de muestras", fontsize=20)
plt.show()

counter_total = Counter(data["deposit"])
print(counter_total)

# Observamos que no tenemos una desproporci�n muy grave con respecto al n�mero de muestras en cada clase, por
# lo tanto, podemos obviar el uso de t�cnicas de sobre muestreo y submuestreo para el rebalanceo de muestras.
# Cabe mencionar que CatBoost tambi�n posee un hiperpar�metro encargado de solucionar este problema, a�adiendo
# pesos a las muestras de la clase minoritaria para que su impacto en el modelo sea casi el mismo que el de la
# clase mayoritaria, por lo tanto, podr�amos hacer uso de esta funci�n para mejorar un poco m�s el rendimiento
# predictivo de nuestro modelo.


#------------------------------------------------------------------------------------------------------------
#                               CONSTRUCCI�N Y EVALUACI�N DEL MODELO PREDICTIVO
#------------------------------------------------------------------------------------------------------------

# Como ya se mencion� en la introducci�n de este proyecto, para la construcci�n de un modelo predictivo
# utilizaremos CatBoost.

# El motivo principal por el que elegimos este algoritmo basado en el aumento del gradiente es porque ofrece
# soporte para el trabajo de clasificaci�n y regresi�n con variables categ�ricas sin la necesidad de pre
# procesarlas, adem�s que en la mayor�a de ocasiones se puede obtener resultados considerablemente buenos sin
# realizar demasiados ajustes en los hiperparametros. Y por �ltimo, porque es relativamente r�pido entrenarlo,
# incluso cuando se tiene una cantidad considerable de datos. Estas cualidades encajan bien con nuestro conjunto
# de datos, puesto que tenemos alrededor de 11000 observaciones las cuales tienen caracter�sticas pertenecientes
# tanto a variables categ�ricas como num�ricas.


#-----------------------------
# ELECCI�N DE HIPERPAR�METROS
#-----------------------------

# Como anteriormente hab�amos dicho, CatBoost puede obtener resultados buenos con la configuraci�n de
# hiperparametros predeterminada, sin embargo, el objetivo de este proyecto es obtener el mejor modelo posible
# que pueda predecir de forma correcta la solicitud de dep�sito a plazo fijo de los clientes, es por ello que
# haciendo uso de la librer�a Optuna, mediante la optimizaci�n bayesiana, intentaremos encontrar la combinaci�n
# de hiperparametros que mejor se ajuste a nuestros datos.

# Dado que a lo largo de este proyecto hemos realizado distintas transformaciones a nuestros datos, y hemos
# guardado una copia del conjunto de datos antes de realizar dicha transformaci�n, aplicaremos la funci�n de
# b�squeda de hiperparametros a cada uno de estos conjuntos, con el fin de comparar hasta que paso de la
# transformaci�n es necesaria para obtener el modelo con el mejor rendimiento posible, o si para este caso, no
# es necesario aplicar transformaci�n alguna. Es por ello que dividiremos esta secci�n en cuatro partes,
# basado en los cuatro conjuntos de datos obtenidos:
    
# Hiperpar�metros para datos con outliers y sin codificaci�n manual
# Hiperpar�metros para datos con outliers y con codificaci�n manual
# Hiperpar�metros para datos sin outliers y sin codificaci�n manual
# Hiperpar�metros para datos sin outliers y con codificaci�n manual

#------------------------------
# HIPERPAR�METROS PARA DATOS CON OUTLIERS Y SIN CODIFICACION MANUAL

def objective(trial):   

    params = {"iterations": trial.suggest_int("iterations",300,1200,100),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
              "depth": trial.suggest_int("depth", 4, 12, 1),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
              "random_strength":trial.suggest_int("random_strength", 0, 40, 1),
              "bagging_temperature": trial.suggest_int("bagging_temperature", 0, 2, 1),
              'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 0, 10),
              "auto_class_weights": "Balanced",
              "loss_function": "Logloss",
              "eval_metric": "AUC",
              "task_type": "GPU",
              "od_type" : "Iter",  # Parametros relacionados con early stop
              "od_wait" : 30,
              "use_best_model": True,
              "random_seed": 42}
    
    # Identificaci�n de variables categoricas
    categorical_features_indices = np.where(data.dtypes == np.object)[0]
    
    train_pool = Pool(X_train, y_train, cat_features = categorical_features_indices)
    test_pool = Pool(X_test, y_test, cat_features = categorical_features_indices)
    
    # Inicializaci�n y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluaci�n y obtenci�n de m�tricas
    preds = model.predict(X_test)
    metric = accuracy_score(y_test, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_1 = study.trials_dataframe()

# Se ejecut� la funci�n tres veces de forma independiente, y posterior a ello, se registr� la mejor combinaci�n
# de hiperpar�metros que arrojo cada ejecuci�n, siendo estas las siguientes:

# 73.96% | iterarions=600, learning_rate=0.166129, depth=7, l2_leaf_reg=0.963535, random_strength=11, bagging_temperature=1, max_ctr_complexity=0
# 74.08% | iterarions=1200, learning_rate=0.246729, depth=9, l2_leaf_reg=7.2024, random_strength=33, bagging_temperature=1, max_ctr_complexity=2
# 73.72% | iterarions=500, learning_rate=0.208787, depth=10, l2_leaf_reg=1.25048, random_strength=15, bagging_temperature=1, max_ctr_complexity=2

# Procederemos a entrenar modelos CatBoost en base a estas tres combinaciones de hiperpar�metros obtenidas para
# determinar cu�l de ellas presenta mejores resultados al clasificar nuestros datos.

# Identificaci�n de variables categoricas
categorical_features_indices1 = np.where(data.dtypes == np.object)[0]
train_pool1 = Pool(X_train, y_train, cat_features = categorical_features_indices1)
test_pool1 = Pool(X_test, y_test, cat_features = categorical_features_indices1)

# Para la primera combinaci�n
cb_1a = CatBoostClassifier(iterations=600, learning_rate=0.166129, depth=7, l2_leaf_reg=0.963535, random_strength=11,
                            bagging_temperature=1, max_ctr_complexity=0, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_1a.fit(train_pool1, eval_set = test_pool1, logging_level="Silent")
y_pred_1a = cb_1a.predict(X_test)

# Para la segunda combinaci�n
cb_1b = CatBoostClassifier(iterations=1200, learning_rate=0.246729, depth=9, l2_leaf_reg=7.2024, random_strength=33,
                            bagging_temperature=1, max_ctr_complexity=2, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_1b.fit(train_pool1, eval_set = test_pool1, logging_level="Silent")
y_pred_1b = cb_1b.predict(X_test)

# Para la tercera combinaci�n
cb_1c = CatBoostClassifier(iterations=500, learning_rate=0.208787, depth=10, l2_leaf_reg=1.25048, random_strength=15,
                            bagging_temperature=1, max_ctr_complexity=2, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_1c.fit(train_pool1, eval_set = test_pool1, logging_level="Silent")
y_pred_1c = cb_1c.predict(X_test)


# COMPARACI�N DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinaci�n
f1_1a = f1_score(y_test, y_pred_1a)
acc_1a = accuracy_score(y_test, y_pred_1a)
auc_1a = roc_auc_score(y_test, y_pred_1a)
report_1a = classification_report(y_test,y_pred_1a)

# Para la segunda combinaci�n
f1_1b = f1_score(y_test, y_pred_1b)
acc_1b = accuracy_score(y_test, y_pred_1b)
auc_1b = roc_auc_score(y_test, y_pred_1b)
report_1b = classification_report(y_test,y_pred_1b)

# Para la tercera combinaci�n
f1_1c = f1_score(y_test, y_pred_1c)
acc_1c = accuracy_score(y_test, y_pred_1c)
auc_1c = roc_auc_score(y_test, y_pred_1c)
report_1c = classification_report(y_test,y_pred_1c)

# A continuaci�n, visualizaremos el puntaje de la m�trica F1 y la precisi�n para cada combinaci�n, a la vez que
# tambi�n observaremos un reporte de las principales m�tricas para evaluar la capacidad de clasificaci�n de
# nuestros modelos.

print("F1 primera comb.: %.2f%%" % (f1_1a * 100.0))
print("Accuracy primera comb.: %.2f%%" % (acc_1a * 100.0))
print("-------------------------------")
print("F1 segunda comb.: %.2f%%" % (f1_1b * 100.0))
print("Accuracy segunda comb.: %.2f%%" % (acc_1b * 100.0))
print("-------------------------------")
print("F1 tercera comb.: %.2f%%" % (f1_1c * 100.0))
print("Accuracy tercera comb.: %.2f%%" % (acc_1c * 100.0))

print(report_1a)
print("-------------------------------------------------")
print(report_1b)
print("-------------------------------------------------")
print(report_1c)

# En principio, observamos que tanto la primera como la segunda combinaci�n presentan valores de m�trica superiores
# en comparaci�n con la tercera combinaci�n, aunque la diferencia entre ellos es m�nima.

# Procederemos a graficar la matriz de confusi�n y la curva ROC-AUC.

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_1a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACI�N 1",fontsize=14)

sns.heatmap(confusion_matrix(y_test, y_pred_1b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACI�N 2",fontsize=14)

sns.heatmap(confusion_matrix(y_test, y_pred_1c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACI�N 3",fontsize=14)

plt.show()

# Con respecto a las matrices de confusi�n, observamos que la segunda combinaci�n presenta un ratio mejor
# equilibrado al momento de predecir correctamente si un cliente solicita un dep�sito a plazo fijo o no, a la
# vez que tambi�n observamos que la tercera combinaci�n presenta resultados ineficientes al realizar esta
# clasificaci�n, ya que predice correctamente m�s muestras positivas (1) en comparaci�n de las dem�s combinaciones,
# pero a coste de una cantidad considerable en la correcta predicci�n de muestras negativas (0). Es por ello
# que tomaremos en cuenta a la segunda combinaci�n como la que mejores resultados arroj� en este apartado de
# evaluaci�n.

y_pred_prob1a = cb_1a.predict_proba(X_test)[:,1]
fpr_1a, tpr_1a, thresholds_1a = roc_curve(y_test, y_pred_prob1a)
y_pred_prob1b = cb_1b.predict_proba(X_test)[:,1]
fpr_1b, tpr_1b, thresholds_1b = roc_curve(y_test, y_pred_prob1b)
y_pred_prob1c = cb_1c.predict_proba(X_test)[:,1]
fpr_1c, tpr_1c, thresholds_1c = roc_curve(y_test, y_pred_prob1c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_1a, tpr_1a, label='Combinaci�n 1',color = "r")
plt.plot(fpr_1b, tpr_1b, label='Combinaci�n 2',color = "g")
plt.plot(fpr_1c, tpr_1c, label='Combinaci�n 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Del grafico de la curva ROC-AUC no podemos diferenciar claramente si la combinaci�n 1 o 2 es la que mejor
# tasa de verdaderos positivos (VP) y falsos positivos (FP) tiene, sin embargo, podemos observar que la curva
# de la tercera combinaci�n tiende a ser menor comparado con las dem�s combinaciones, por lo que combinado con
# los resultados de las m�tricas anteriores, podemos ir descartando esta combinaci�n.

print("AUC primera comb.: %.2f%%" % (auc_1a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_1b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_1c * 100.0))

# Por �ltimo, podemos ver que el valor de la m�trica AUC nos da claridad al momento de decidir qu� combinaci�n
# presenta una mejor tasa de VP y FP, ya que como hab�amos deducido anteriormente, la tercera combinaci�n es la
# que peores resultados arroja, y que tanto la primera como la segunda combinaci�n presentan resultados
# similares, sin embargo, la segunda combinaci�n presenta una ligera superioridad comparado con las dem�s
# combinaciones. Entonces, uniendo los resultados de las m�tricas anteriormente vistas, podemos concluir que el
# modelo construido con la segunda combinaci�n es el que mejor clasifica estos datos, por lo tanto, utilizaremos
# este modelo como referente del conjunto de "Hiperpar�metros para datos con outliers y sin codificaci�n manual".
 

#------------------------------
# HIPERPAR�METROS PARA DATOS CON OUTLIERS Y CON CODIFICACI�N MANUAL

def objective(trial):   

    params = {"iterations": trial.suggest_int("iterations",300,1200,100),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
              "depth": trial.suggest_int("depth", 4, 12, 1),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
              "random_strength":trial.suggest_int("random_strength", 0, 40, 1),
              "bagging_temperature": trial.suggest_int("bagging_temperature", 0, 2, 1),
              'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 0, 10),
              "auto_class_weights": "Balanced",
              "loss_function": "Logloss",
              "eval_metric": "AUC",
              "task_type": "GPU",
              "od_type" : "Iter",
              "od_wait" : 30,
              "use_best_model": True,
              "random_seed": 42}
    
    train_pool = Pool(X_train_cod, y_train_cod)
    test_pool = Pool(X_test_cod, y_test_cod)
    
    # Inicializaci�n y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluaci�n y obtenci�n de m�tricas
    preds = model.predict(X_test_cod)
    metric = accuracy_score(y_test_cod, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_2 = study.trials_dataframe()

# Se ejecut� la funci�n tres veces de forma independiente, y posterior a ello, se registro
# la mejor combinaci�n de par�metros que arrojo cada ejecuci�n, siendo estas las siguientes:

# 74.29% | iterarions=900, learning_rate=0.198706, depth=9, l2_leaf_reg=4.72514, random_strength=40, bagging_temperature=0, max_ctr_complexity=3
# 74.23% | iterarions=1000, learning_rate=0.0686307, depth=7, l2_leaf_reg=6.87847, random_strength=2, bagging_temperature=1, max_ctr_complexity=1
# 74.17% | iterarions=500, learning_rate=0.103597, depth=11, l2_leaf_reg=7.95198, random_strength=8, bagging_temperature=0, max_ctr_complexity=4

# Procederemos a entrenar modelos CatBoost en base a estas tres combinaciones de hiperpar�metros obtenidas para
# determinar cu�l de ellas presenta mejores resultados al clasificar nuestros datos.

train_pool2 = Pool(X_train_cod, y_train_cod)
test_pool2 = Pool(X_test_cod, y_test_cod)

# Para la primera combinaci�n
cb_2a = CatBoostClassifier(iterations=900, learning_rate=0.198706, depth=9, l2_leaf_reg=4.72514, random_strength=40,
                            bagging_temperature=0, max_ctr_complexity=3, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_2a.fit(train_pool2, eval_set = test_pool2, logging_level="Silent")
y_pred_2a = cb_2a.predict(X_test_cod)

# Para la segunda combinaci�n
cb_2b = CatBoostClassifier(iterations=1000, learning_rate=0.0686307, depth=7, l2_leaf_reg=6.87847, random_strength=2,
                            bagging_temperature=1,  max_ctr_complexity=1, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_2b.fit(train_pool2, eval_set = test_pool2, logging_level="Silent")
y_pred_2b = cb_2b.predict(X_test_cod)

# Para la tercera combinaci�n
cb_2c = CatBoostClassifier(iterations=500, learning_rate=0.103597, depth=11, l2_leaf_reg=7.95198, random_strength=8,
                            bagging_temperature=0, max_ctr_complexity=4, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_2c.fit(train_pool2, eval_set = test_pool2, logging_level="Silent")
y_pred_2c = cb_2c.predict(X_test_cod)


# COMPARACI�N DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinaci�n
f1_2a = f1_score(y_test_cod, y_pred_2a)
acc_2a = accuracy_score(y_test_cod, y_pred_2a)
auc_2a = roc_auc_score(y_test_cod, y_pred_2a)
report_2a = classification_report(y_test_cod,y_pred_2a)

# Para la segunda combinaci�n
f1_2b = f1_score(y_test_cod, y_pred_2b)
acc_2b = accuracy_score(y_test_cod, y_pred_2b)
auc_2b = roc_auc_score(y_test_cod, y_pred_2b)
report_2b = classification_report(y_test_cod,y_pred_2b)

# Para la tercera combinaci�n
f1_2c = f1_score(y_test_cod, y_pred_2c)
acc_2c = accuracy_score(y_test_cod, y_pred_2c)
auc_2c = roc_auc_score(y_test_cod, y_pred_2c)
report_2c = classification_report(y_test_cod,y_pred_2c)


print("F1 primera comb.: %.2f%%" % (f1_2a * 100.0))
print("Accuracy primera comb.: %.2f%%" % (acc_2a * 100.0))
print("-------------------------------")
print("F1 segunda comb.: %.2f%%" % (f1_2b * 100.0))
print("Accuracy segunda comb.: %.2f%%" % (acc_2b * 100.0))
print("-------------------------------")
print("F1 tercera comb.: %.2f%%" % (f1_2c * 100.0))
print("Accuracy tercera comb.: %.2f%%" % (acc_2c * 100.0))

print(report_2a)
print("-------------------------------------------------")
print(report_2b)
print("-------------------------------------------------")
print(report_2c)

# Se puede observar que, si bien todos los valores de m�trica para todas las combinaciones que tenemos son muy
# similares, la tercera combinaci�n es la que destaca por un peque�o margen porcentual de las dem�s. Mientras
# que tanto la primera como la segunda combinaci�n parecen tener un rendimiento equivalente.

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACI�N 1",fontsize=14)

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACI�N 2",fontsize=14)

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACI�N 3",fontsize=14)

plt.show()

# De las matrices de confusi�n podemos reafirmar nuestra suposici�n al observar que la tercera combinaci�n es
# la que presenta un mejor ratio de VP y FP, mientras que la segunda y la primera combinaci�n presentan ratios
# que se complementan entre s�, ya que el punto fuerte de la primera combinaci�n son los VP, mientras que su
# punto d�bil, los FP, es el punto fuerte de la segunda combinaci�n.

y_pred_prob2a = cb_2a.predict_proba(X_test_cod)[:,1]
fpr_2a, tpr_2a, thresholds_2a = roc_curve(y_test_cod, y_pred_prob2a)
y_pred_prob2b = cb_2b.predict_proba(X_test_cod)[:,1]
fpr_2b, tpr_2b, thresholds_2b = roc_curve(y_test_cod, y_pred_prob2b)
y_pred_prob2c = cb_2c.predict_proba(X_test_cod)[:,1]
fpr_2c, tpr_2c, thresholds_2c = roc_curve(y_test_cod, y_pred_prob2c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_2a, tpr_2a, label='Combinaci�n 1',color = "r")
plt.plot(fpr_2b, tpr_2b, label='Combinaci�n 2',color = "g")
plt.plot(fpr_2c, tpr_2c, label='Combinaci�n 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Con respecto a la curva ROC-AUC no se puede visualizar una clara diferencia entre las curvas de la combinaci�n
# 2 y 3, sin embargo, se puede observar que la curva de la primera combinaci�n parece estar por debajo en
# comparaci�n con el de las dem�s combinaciones, lo cual nos dice que no tiene un buen ratio compar�ndola con
# las dem�s al momento de clasificar una muestra con la clase que le corresponde (VP, FP).

print("AUC primera comb.: %.2f%%" % (auc_2a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_2b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_2c * 100.0))

# Por �ltimo, con un valor porcentual de la m�trica de la curva, podemos tomar una decisi�n final respecto a
# que combinaci�n elegir. Tomar esta decisi�n no ser� muy dif�cil, ya que en 3 de las 4 pruebas de evaluaci�n
# observamos claramente que la tercera combinaci�n es la que present� mejores resultados en comparaci�n con las
# dem�s combinaciones, aunque la diferencia entre estas sea porcentual, es por ello que el modelo construido
# con esta combinaci�n ser� usado como referente del conjunto de "Hiperparametros para datos con outliers y
# con codificaci�n manual".


#------------------------------
# HIPERPAR�METROS PARA DATOS SIN OUTLIERS Y SIN CODIFICACI�N MANUAL

def objective(trial):   

    params = {"iterations": trial.suggest_int("iterations",300,1200,100),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
              "depth": trial.suggest_int("depth", 4, 12, 1),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
              "random_strength":trial.suggest_int("random_strength", 0, 40, 1),
              "bagging_temperature": trial.suggest_int("bagging_temperature", 0, 2, 1),
              'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 0, 10),
              "auto_class_weights": "Balanced",
              "loss_function": "Logloss",
              "eval_metric": "AUC",
              "task_type": "GPU",
              "od_type" : "Iter",
              "od_wait" : 30,
              "use_best_model": True,
              "random_seed": 42}
    
    # Identificaci�n de variables categoricas
    categorical_features_indices = np.where(data2.dtypes == np.object)[0]
    
    train_pool = Pool(X2_train, y2_train, cat_features = categorical_features_indices)
    test_pool = Pool(X2_test, y2_test, cat_features = categorical_features_indices)
    
    # Inicializaci�n y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluaci�n y obtenci�n de m�tricas
    preds = model.predict(X2_test)
    metric = accuracy_score(y2_test, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_3 = study.trials_dataframe()

# Se ejecut� la funci�n tres veces de forma independiente, y posterior a ello, se registr� la mejor combinaci�n
# de par�metros que arroj� cada ejecuci�n, siendo estas las siguientes:

# 75.84% | iterarions=900, learning_rate=0.118849, depth=7, l2_leaf_reg=9.48661, random_strength=15, bagging_temperature=1, max_ctr_complexity=6
# 75.72% | iterarions=1000, learning_rate=0.115247, depth=10, l2_leaf_reg=7.87387, random_strength=19, bagging_temperature=1, max_ctr_complexity=6
# 75.66% | iterarions=700, learning_rate=0.0431437, depth=10, l2_leaf_reg=7.91287, random_strength=14, bagging_temperature=0, max_ctr_complexity=4

# Procederemos a entrenar modelos CatBoost en base a estas tres combinaciones de hiperpar�metros obtenidas para
# determinar cu�l de ellas presenta mejores resultados al clasificar nuestros datos.

# Identificaci�n de variables categ�ricas
categorical_features_indices3 = np.where(data2.dtypes == np.object)[0]
train_pool3 = Pool(X2_train, y2_train, cat_features = categorical_features_indices3)
test_pool3 = Pool(X2_test, y2_test, cat_features = categorical_features_indices3)

# Para la primera combinaci�n
cb_3a = CatBoostClassifier(iterations=900, learning_rate=0.118849, depth=7, l2_leaf_reg=9.48661, random_strength=15,
                            bagging_temperature=1, max_ctr_complexity= 6, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_3a.fit(train_pool3, eval_set = test_pool3, logging_level="Silent")
y_pred_3a = cb_3a.predict(X2_test)

# Para la segunda combinaci�n
cb_3b = CatBoostClassifier(iterations=1000, learning_rate=0.115247, depth=10, l2_leaf_reg=7.87387, random_strength=19,
                            bagging_temperature=1,  max_ctr_complexity= 6, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_3b.fit(train_pool3, eval_set = test_pool3, logging_level="Silent")
y_pred_3b = cb_3b.predict(X2_test)

# Para la tercera combinaci�n
cb_3c = CatBoostClassifier(iterations=700, learning_rate=0.0431437, depth=10, l2_leaf_reg=7.91287, random_strength=14,
                            bagging_temperature=0, max_ctr_complexity= 4, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_3c.fit(train_pool3, eval_set = test_pool3, logging_level="Silent")
y_pred_3c = cb_3c.predict(X2_test)


# COMPARACI�N DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinaci�n
f1_3a = f1_score(y2_test, y_pred_3a)
acc_3a = accuracy_score(y2_test, y_pred_3a)
auc_3a = roc_auc_score(y2_test, y_pred_3a)
report_3a = classification_report(y2_test,y_pred_3a)

# Para la segunda combinaci�n
f1_3b = f1_score(y2_test, y_pred_3b)
acc_3b = accuracy_score(y2_test, y_pred_3b)
auc_3b = roc_auc_score(y2_test, y_pred_3b)
report_3b = classification_report(y2_test,y_pred_3b)

# Para la tercera combinaci�n
f1_3c = f1_score(y2_test, y_pred_3c)
acc_3c = accuracy_score(y2_test, y_pred_3c)
auc_3c = roc_auc_score(y2_test, y_pred_3c)
report_3c = classification_report(y2_test,y_pred_3c)


print("F1 primera comb.: %.2f%%" % (f1_3a * 100.0))
print("Accuracy primera comb.: %.2f%%" % (acc_3a * 100.0))
print("-------------------------------")
print("F1 segunda comb.: %.2f%%" % (f1_3b * 100.0))
print("Accuracy segunda comb.: %.2f%%" % (acc_3b * 100.0))
print("-------------------------------")
print("F1 tercera comb.: %.2f%%" % (f1_3c * 100.0))
print("Accuracy tercera comb.: %.2f%%" % (acc_3c * 100.0))

print(report_3a)
print("-------------------------------------------------")
print(report_3b)
print("-------------------------------------------------")
print(report_3c)

# Observamos que la primera combinaci�n tiene un rendimiento inferior en comparaci�n con las dem�s combinaciones,
# y que existe mucha similitud entre los puntajes obtenidos por las combinaciones 2 y 3, donde la combinaci�n
# que sobresale en comparaci�n con la otra en puntaje F1, no lo hace en Accuracy, y la que lo hace en Accuracy,
# no lo hace en F1, es por ello que necesitamos m�s indicios (m�tricas de evaluaci�n) que nos indiquen que
# combinaci�n se adapta mejor a nuestros datos.

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y2_test, y_pred_3a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACI�N 1",fontsize=14)

sns.heatmap(confusion_matrix(y2_test, y_pred_3b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACI�N 2",fontsize=14)

sns.heatmap(confusion_matrix(y2_test, y_pred_3c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACI�N 3",fontsize=14)

plt.show()

# De las matrices de confusi�n observamos que la segunda combinaci�n es el que mejor ratio en VP tiene, sin
# embargo, es muy inferior en comparaci�n con las dem�s combinaciones con respecto a los FP. Un comportamiento
# similar se observa en la tercera combinaci�n, donde la situaci�n es la misma pero invertida, donde el mejor
# ratio de predicci�n se lo llevan los FP y el peor los VP. Sin embargo, ya que su ratio de VP es ligeramente
# mayor al de la primera combinaci�n, e inferior al de la segunda combinaci�n, podemos decir que se encuentra
# en t�rmino medio, y esto en conjunto con su indiscutible superioridad en la predicci�n de FP, lo hace un
# modelo mejor en comparaci�n con los dem�s, es por ello que en el �rea de las matrices de confusi�n preferimos
# eta combinaci�n.

y_pred_prob3a = cb_3a.predict_proba(X2_test)[:,1]
fpr_3a, tpr_3a, thresholds_3a = roc_curve(y2_test, y_pred_prob3a)
y_pred_prob3b = cb_3b.predict_proba(X2_test)[:,1]
fpr_3b, tpr_3b, thresholds_3b = roc_curve(y2_test, y_pred_prob3b)
y_pred_prob3c = cb_3c.predict_proba(X2_test)[:,1]
fpr_3c, tpr_3c, thresholds_3c = roc_curve(y2_test, y_pred_prob3c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_3a, tpr_3a, label='Combinaci�n 1',color = "r")
plt.plot(fpr_3b, tpr_3b, label='Combinaci�n 2',color = "g")
plt.plot(fpr_3c, tpr_3c, label='Combinaci�n 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Con respecto a la curva ROC-AUC podemos identificar que aparentemente las combinaciones 2 y 3 son las que
# presentan un mejor ratio en VP y FP, sin embargo, la diferencia entre estas no es tan clara, puesto que hay
# momentos en donde presentan una inclinaci�n similar, y otros en donde una muestra superioridad sobre la otra,
# es por ello que calcularemos el valor de su m�trica con el fin de poder tener una mejor interpretabilidad.

print("AUC primera comb.: %.2f%%" % (auc_3a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_3b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_3c * 100.0))

# Con esto �ltimo, observamos que el mejor valor AUC se lo lleva la tercera combinaci�n, el peor valor la primera
# combinaci�n, y en t�rmino medio se encuentra la segunda combinaci�n. No obstante, la diferencia entre estos
# valores es muy peque�a, por lo cual no podemos decir que una combinaci�n es mucho m�s efectiva que otra.
# Entonces, uniendo los resultados de todas las m�tricas vistas, podemos considerar a todas las combinaciones
# como buenas, ya que no existe una gran diferencia en efectividad de predicci�n entre ellas, sin embargo, en
# esta ocasi�n elegiremos la tercera combinaci�n como ganadora, ya que es la que presenta un equilibrio entre
# predecir correctamente aquellas muestras que son positivas (dep�sito a plazo fijo) y negativas (no dep�sito
# a plazo fijo), por lo tanto, utilizaremos el modelo construido con esta combinaci�n como referente del
# conjunto de "Hiperparametros para datos sin outliers y sin codificaci�n manual".


#------------------------------
# HIPERPAR�METROS PARA DATOS SIN OUTLIERS Y CON CODIFICACI�N MANUAL

def objective(trial):   

    params = {"iterations": trial.suggest_int("iterations",300,1200,100),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
              "depth": trial.suggest_int("depth", 4, 12, 1),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
              "random_strength":trial.suggest_int("random_strength", 0, 40, 1),
              "bagging_temperature": trial.suggest_int("bagging_temperature", 0, 2, 1),
              'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 0, 10),
              "auto_class_weights": "Balanced",
              "loss_function": "Logloss",
              "eval_metric": "AUC",
              "task_type": "GPU",
              "od_type" : "Iter",
              "od_wait" : 30,
              "use_best_model": True,
              "random_seed": 42}
    
    train_pool = Pool(X2_train_cod, y2_train_cod)
    test_pool = Pool(X2_test_cod, y2_test_cod)
    
    # Inicializaci�n y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluaci�n y obtenci�n de m�tricas
    preds = model.predict(X2_test_cod)
    metric = accuracy_score(y2_test_cod, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_4 = study.trials_dataframe()

# Se ejecut� la funci�n tres veces de forma independiente, y posterior a ello, se registr� la mejor combinaci�n
# de par�metros que arroj� cada ejecuci�n, siendo estas las siguientes:

# 75.24% | iterarions=600, learning_rate=0.0856975, depth=9, l2_leaf_reg=7.42101, random_strength=0, bagging_temperature=0, max_ctr_complexity=10
# 75.18% | iterarions=1200, learning_rate=0.0274008, depth=10, l2_leaf_reg=7.42817, random_strength=0, bagging_temperature=1, max_ctr_complexity=1
# 75.15% | iterarions=700, learning_rate=0.0854912, depth=10, l2_leaf_reg=5.43813, random_strength=0, bagging_temperature=0, max_ctr_complexity=10

# Procederemos a entrenar un nuevo modelo XGBoost en base a las tres combinaciones de hiperpar�metros
# obtenidas para determinar cual de ellas presenta mejores resultados al clasificar nuestros datos

train_pool4 = Pool(X2_train_cod, y2_train_cod)
test_pool4 = Pool(X2_test_cod, y2_test_cod)

# Para la primera combinaci�n
cb_4a = CatBoostClassifier(iterations=600, learning_rate=0.0856975, depth=9, l2_leaf_reg=7.42101, random_strength=0,
                            bagging_temperature=0, max_ctr_complexity=10, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_4a.fit(train_pool4, eval_set = test_pool4, logging_level="Silent")
y_pred_4a = cb_4a.predict(X2_test_cod)

# Para la segunda combinaci�n
cb_4b = CatBoostClassifier(iterations=1200, learning_rate=0.0274008, depth=10, l2_leaf_reg=7.42817, random_strength=0,
                            bagging_temperature=1,  max_ctr_complexity=1, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_4b.fit(train_pool4, eval_set = test_pool4, logging_level="Silent")
y_pred_4b = cb_4b.predict(X2_test_cod)

# Para la tercera combinaci�n
cb_4c = CatBoostClassifier(iterations=700, learning_rate=0.0854912, depth=10, l2_leaf_reg=5.43813, random_strength=0,
                            bagging_temperature=0, max_ctr_complexity=10, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_4c.fit(train_pool4, eval_set = test_pool4, logging_level="Silent")
y_pred_4c = cb_4c.predict(X2_test_cod)


# COMPARACI�N DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinaci�n
f1_4a = f1_score(y2_test_cod, y_pred_4a)
acc_4a = accuracy_score(y2_test_cod, y_pred_4a)
auc_4a = roc_auc_score(y2_test_cod, y_pred_4a)
report_4a = classification_report(y2_test_cod,y_pred_4a)

# Para la segunda combinaci�n
f1_4b = f1_score(y2_test_cod, y_pred_4b)
acc_4b = accuracy_score(y2_test_cod, y_pred_4b)
auc_4b = roc_auc_score(y2_test_cod, y_pred_4b)
report_4b = classification_report(y2_test_cod,y_pred_4b)

# Para la tercera combinaci�n
f1_4c = f1_score(y2_test_cod, y_pred_4c)
acc_4c = accuracy_score(y2_test_cod, y_pred_4c)
auc_4c = roc_auc_score(y2_test_cod, y_pred_4c)
report_4c = classification_report(y2_test_cod,y_pred_4c)


print("F1 primera comb.: %.2f%%" % (f1_4a * 100.0))
print("Accuracy primera comb.: %.2f%%" % (acc_4a * 100.0))
print("-------------------------------")
print("F1 segunda comb.: %.2f%%" % (f1_4b * 100.0))
print("Accuracy segunda comb.: %.2f%%" % (acc_4b * 100.0))
print("-------------------------------")
print("F1 tercera comb.: %.2f%%" % (f1_4c * 100.0))
print("Accuracy tercera comb.: %.2f%%" % (acc_4c * 100.0))

print(report_4a)
print("-------------------------------------------------")
print(report_4b)
print("-------------------------------------------------")
print(report_4c)

# Respecto a esta combinaci�n de m�tricas se observa claramente que la primera combinaci�n es la que mejores
# resultados presenta, aunque la diferencia de sus puntajes comparados con el de las dem�s combinaciones sea
# relativamente peque�a.

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACI�N 1",fontsize=14)

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACI�N 2",fontsize=14)

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACI�N 3",fontsize=14)

plt.show()

# De las matrices de confusi�n observamos que todas las combinaciones tienen resultados muy similares los unos
# con los otros, por lo tanto, es dif�cil decidir en este apartado de evaluaci�n que combinaci�n resulta m�s
# efectiva en la predicci�n clientes solicitantes y no solicitantes de un dep�sito a plazo fijo.

y_pred_prob4a = cb_4a.predict_proba(X2_test_cod)[:,1]
fpr_4a, tpr_4a, thresholds_4a = roc_curve(y2_test_cod, y_pred_prob4a)
y_pred_prob4b = cb_4b.predict_proba(X2_test_cod)[:,1]
fpr_4b, tpr_4b, thresholds_4b = roc_curve(y2_test_cod, y_pred_prob4b)
y_pred_prob4c = cb_4c.predict_proba(X2_test_cod)[:,1]
fpr_4c, tpr_4c, thresholds_4c = roc_curve(y2_test_cod, y_pred_prob4c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_4a, tpr_4a, label='Combinaci�n 1',color = "r")
plt.plot(fpr_4b, tpr_4b, label='Combinaci�n 2',color = "g")
plt.plot(fpr_4c, tpr_4c, label='Combinaci�n 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Con respecto a la curva ROC-AUC, no podemos identificar claramente que combinaci�n es superior a las dem�s,
# ya que existen algunos trazos en los que una combinaci�n es inferior a otra, y otros en los que es superior
# a las dem�s, es por ello que calcularemos el valor de su m�trica con el fin de poder tener una mejor
# interpretabilidad.

print("AUC primera comb.: %.2f%%" % (auc_4a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_4b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_4c * 100.0))

# Estos resultados indican que la primera combinaci�n posee un mejor ratio en la correcta predicci�n de nuestros
# datos, sin embargo, cabe recalcar que la diferencia de este valor comparado con el de las dem�s combinaciones
# es muy peque�a, es por ello que elegir cualquiera de estas combinaciones presentadas ser�a una opci�n correcta,
# ya que todas tienen un rendimiento muy similar. Para este caso, bas�ndonos en la diferencia porcentual de
# valores de m�trica ya vistos, elegiremos el modelo construido con la primera combinaci�n como el que mejor
# clasifica nuestros datos, por lo tanto, ser� usado como referente del conjunto de "Hiperparametros para datos
# sin outliers y con codificaci�n manual".


#-----------------------------
# ELECCI�N DEL MEJOR MODELO
#-----------------------------

# Despu�s de haber elegido las cuatro mejores combinaciones en base al entrenamiento de conjuntos con diferentes
# tipos de transformaci�n y codificaci�n, procederemos a compararlos entre s� para quedarnos con un modelo
# definitivo el cual mejores resultados de evaluaci�n tenga.

print("F1 Primer conjunto: %.2f%%" % (f1_1b * 100.0))
print("Accuracy Primer conjunto: %.2f%%" % (acc_1b * 100.0))
print("-------------------------------")
print("F1 Segundo conjunto: %.2f%%" % (f1_2c * 100.0))
print("Accuracy Segundo conjunto: %.2f%%" % (acc_2c * 100.0))
print("-------------------------------")
print("F1 Tercer conjunto: %.2f%%" % (f1_3c * 100.0))
print("Accuracy Tercer conjunto: %.2f%%" % (acc_3c * 100.0))
print("-------------------------------")
print("F1 Cuarto conjunto: %.2f%%" % (f1_4a * 100.0))
print("Accuracy Cuarto conjunto: %.2f%%" % (acc_4a * 100.0))

print(report_1b)
print("-------------------------------------------------")
print(report_2c)
print("-------------------------------------------------")
print(report_3c)
print("-------------------------------------------------")
print(report_4a)

# En principio observamos que el modelo construido con el tercer conjunto (Datos sin outliers y sin codificaci�n
# manual) presenta un rendimiento superior en cuanto a Accuracy se refiere, sin embargo, en cuanto a puntaje F1,
# es inferior al modelo construido con el cuarto conjunto (Datos sin outliers y con codificaci�n manual). Los
# conjuntos de datos que si presentan outliers parecen tener un rendimiento inferior en comparaci�n de los
# conjuntos a los cuales si se les realizo un tratamiento de estos valores.

fig, ax = plt.subplots(2, 2, figsize=(20, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_1b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0,0])
ax[0][0].set_title("PRIMER CONJUNTO",fontsize=14)

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0,1])
ax[0][1].set_title("SEGUNDO CONJUNTO",fontsize=14)

sns.heatmap(confusion_matrix(y2_test, y_pred_3c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1,0])
ax[1][0].set_title("TERCER CONJUNTO",fontsize=14)

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1,1])
ax[1][1].set_title("CUARTO CONJUNTO",fontsize=14)

plt.show()

# Las matrices de confusi�n nos dejan ver que la competencia en la elecci�n del mejor modelo predictivo se
# encuentra entre los modelos construidos con el tercer y cuarto conjunto (Datos sin outliers), ya que presentan
# superioridad predictora en comparaci�n con los modelos construidos con el primer y segundo conjunto (Datos
# con outliers). En cuanto al tercer conjunto, podemos observar que su punto fuerte es la predicci�n de muestras
# catalogadas con clase negativa (0), y en cuanto al cuarto conjunto, su punto fuerte es la predicci�n de muestras
# catalogadas con clase positiva (1). Dependiendo de las necesidades de la empresa en tomar m�s atenci�n a
# aquellos clientes que soliciten o no soliciten un dep�sito a plazo fijo, se podr�a elegir cualquier de estos
# dos modelos como valido.

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_1b, tpr_1b, label='Primer conjunto',color = "r")
plt.plot(fpr_2c, tpr_2c, label='Segundo conjunto',color = "g")
plt.plot(fpr_3c, tpr_3c, label='Tercer conjunto',color = "b")
plt.plot(fpr_4a, tpr_4a, label='Cuarto conjunto',color = "y")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Con respecto a la curva ROC-AUC, observamos una clara similitud entre los modelos construidos con el tercer y
# cuarto conjunto, como anteriormente hab�amos visto, estos modelos presentan superioridad predictiva en
# comparaci�n con los construidos por el primer y segundo conjunto, lo cual nos da m�s motivos para ir
# concluyendo que estos modelos no ser�n elegidos como definitivos. Para poder tener una visi�n m�s clara acerca
# de qu� conjunto (tercero o cuarto) presenta una mejor sensibilidad predictora, calcularemos el valor de la
# m�trica AUC.

print("AUC Primer conjunto: %.2f%%" % (auc_1b * 100.0))
print("AUC Segundo conjunto: %.2f%%" % (auc_2c * 100.0))
print("AUC Tercer conjunto: %.2f%%" % (auc_3c * 100.0))
print("AUC Cuarto conjunto: %.2f%%" % (auc_4a * 100.0))

# Estos resultados indican que el modelo construido con el tercer conjunto presenta un ratio mayor en la
# predicci�n de VP y FP en comparaci�n con los dem�s conjuntos, aunque la diferencia entre este conjunto y el
# cuarto es m�nima. Entonces, uniendo los resultados de las dem�s m�tricas de evaluaci�n anteriormente vistas,
# podemos llegas a concluir que tanto el modelo construido con el tercer conjunto como con el cuarto, son
# completamente v�lidos para ser considerados como definitivos, y que su elecci�n depende mucho del objetivo
# que tenga el cliente o la empresa en cuanto a que clase necesita enfocar la predicci�n. En esta ocasi�n,
# tomaremos el modelo construido con el tercer conjunto como definitivo, ya que presenta una ligera superioridad
# en cuanto a AUC Y Accuracy en comparaci�n con los dem�s modelos.

# Combinacion de parametros del modelo final:

# iterarions=700, learning_rate=0.0431437, depth=10, l2_leaf_reg=7.91287, random_strength=14, bagging_temperature=0, max_ctr_complexity=4

# Guardado del modelo
joblib.dump(cb_3c, "CatBoost_Model_BankMarketing")


#-----------------------------
# INTERPRETACI�N DEL MODELO
#-----------------------------

# Esta secci�n estar� dedicada a entender como las variables y los posibles valores que estas tomen afectan en
# la decisi�n del modelo sobre clasificar a un cliente como solicitante o no solicitante de un dep�sito a plazo
# fijo. Para ello, haremos uso de la funci�n de extracci�n de importancia de caracter�sticas implementada por
# CatBoost, y de los valores SHAP basados en la teor�a de juegos.

# IMPORTANCIA DE CARACTER�STICAS DE CATBOOST
#-------------------------------------------

columns = data2.iloc[:,:-1].columns.values  # Extracci�n de los nombres de las variables independientes

pd.DataFrame({'Importancia': cb_3c.get_feature_importance(), 'Variable': columns}).sort_values(by=['Importancia'],
                                                                                               ascending=False)


# El marco de datos obtenido nos dice que la variable "month" tiene un papel fundamental en la clasificaci�n
# de una muestra como solicitante o no solicitante de un dep�sito a plazo fijo. Esto tiene sentido, ya que en
# la secci�n del an�lisis de datos hab�amos observado que exist�an algunos meses en los que era muy probable
# que el cliente aceptase solicitar este tipo de dep�sito (marzo, septiembre, octubre y diciembre) y otros en
# los que no (mayo). Por otra parte, tambi�n observamos que variables como "poutcome" y "contact" juegan un
# papel crucial en dicha predicci�n, puesto que aquellos clientes que hab�an solicitado este dep�sito en la
# campa�a anterior, con mucha probabilidad volver�an a solicitarlo en la campa�a actual, mientras que aquellos
# cuyo medio de contacto era desconocido por el banco, ten�an menos probabilidades de solicitar este dep�sito.
# A la vez que tambi�n se puede apreciar que variables como "default" (mora crediticia con el banco) o
# "previous" (n�mero de contactos realizados en la anterior campa�a) aportan poco en la decisi�n final del modelo.


# VALORES SHAP
#--------------

X2_train_shap = pd.DataFrame(X2_train)  # Conversi�n del conjunto de entrenamiento en Dataframe
X2_train_shap.columns = columns  # Extracci�n de los nombres de las variables independientes

# Codificaci�n del tipo de variable de Object a Int
X2_train_shap[["age", "balance", "day", "campaign", "previous"]] = X2_train_shap[["age", "balance", "day", "campaign",
                                                                                  "previous"]].astype(int)


shap_values = cb_3c.get_feature_importance(train_pool3, type='ShapValues')  # Calculo de los valores SHAP 
expected_value = shap_values[0,-1]  # Extracci�n de los valores SHAP de la columna dependiente
shap_values = shap_values[:,:-1]  # Extracci�n de los valores SHAP de las columnas independientes


# Empezaremos por calcular los valores SHAP para las variables de una muestra determinada en nuestro conjunto
# de datos.

shap.initjs()
shap.force_plot(expected_value, shap_values[0,:], X2_train_shap.iloc[0,:])

# Observamos que la probabilidad de que un cliente solicite un dep�sito a plazo fijo aumenta para esta muestra
# cuando su ocupaci�n (job) es ser estudiante, se le contacto en el mes de mayo (month), el tipo de contacto
# realizado fue por celular (contact) y solicito este mismo servicio en la campa�a anterior (poutcome),
# mientras que sus probabilidades decrecen si es que este solicito un pr�stamo de vivienda (housing). Estos
# hechos respaldan los insights que hab�amos obtenido en la secci�n de an�lisis de datos, donde pudimos
# identificar que los clientes que eran estudiantes o jubilados eran propensos a solicitar este dep�sito, a la
# vez que si estos hab�an aceptado solicitar este servicio en la campa�a anterior, con mucha probabilidad volver�an
# a hacerlo en la campa�a actual, es por ello que en el gr�fico, "poutcome = success" tiene un mayor impacto en
# el impulso del modelo hacia predecir a este cliente como solicitante de un dep�sito a plazo fijo. Mientras
# que por su parte, "housing = yes", tiene un impulso para predecir a este cliente como no solicitante de este
# tipo de dep�sito.

shap.force_plot(expected_value, shap_values[100,:], X2_train_shap.iloc[100,:])

# Por otra parte, en este ejemplo observamos que existen muchas m�s variables con valores que propician al
# modelo a tomar la decisi�n de que cliente no ser� solicitante de un dep�sito a plazo fijo, ya que volviendo
# a recalcar, como anteriormente hab�amos visto en la secci�n de an�lisis de datos, cuando el tipo de contacto
# que se ten�a con el cliente era desconocido por el banco, o cuando no se ten�a informaci�n acerca si este
# hab�a aceptado solicitar este tipo de dep�sito en la campa�a anterior, sus probabilidades de aceptar solicitar
# el dep�sito en la campa�a actual se ve�an reducidas, es por ello que este comportamiento se ve reflejado en
# el gr�fico de SHAP mediante barras color azul en direcci�n a la izquierda.

shap.force_plot(expected_value, shap_values[1000,:], X2_train_shap.iloc[1000,:])

# Esta muestra presenta un comportamiento similar al de la primera, en donde aquellas variables que impulsan al
# modelo a tomar la decisi�n de clasificar a este cliente como solicitante de un dep�sito a plazo fijo son
# "poutcome", "contact", "month" y "job", cuando estas adquieren un valor de "succes", "cellular", "may" y
# "blue-collar" respectivamente. Mientras que las variables que impulsan al modelo a tomar la decisi�n de
# clasificarlo como no solicitante son "housing" y "campaign", cuando estas adquieren un valor de "yes" y "6"
# respectivamente. Algunos de estos valores pueden dar la ilusi�n de contradecir los an�lisis anteriormente
# vistos, como en el caso de "month = may" donde pudimos observar que en este mes hab�an m�s posibilidades de
# que el cliente no aceptara solicitar este tipo de dep�sito, sin embargo, debido a que estamos hablando de
# posibilidad y no de hechos, se asume que aunque exista una alta de probabilidad de rechazar el dep�sito en
# este mes, a�n existen probabilidades de no hacerlo, es por ello que el grafico se muestra a "month = may"
# como una variable que influenci� (aunque en peque�a medida) a la decisi�n del modelo a clasificar a este cliente como solicitante de un dep�sito a plazo fijo.

# Por �ltimo, graficaremos un resumen general del impacto de los valores que pueden tomar cada una de nuestras
# variables en la decisi�n del modelo. Cabe mencionar que actualmente esta gr�fica solo funciona para variable
# num�ricas, puesto que son las �nicas a las que se le puede asociar valores altos y bajos.

shap.summary_plot(shap_values, X2_train_shap)

# Este grafico aparte de mostrar en orden descendente la importancia de cada variable para la predicci�n del
# modelo, nos muestra c�mo afectan tanto los valores altos como los bajos en su decisi�n de clasificar a un
# cliente como solicitante o no solicitante de un dep�sito a plazo fijo. Donde podemos observar que un saldo
# (balance) bajo en la cuenta del cliente influye a que el modelo lo catalogue como no solicitante, o que una
# edad (age) mayor influye a que este lo catalogue como solicitante.

# Todos estos gr�ficos son muy importantes, ya que nos proporcionan insights muy �tiles para entender a mayor
# profundidad el comportamiento o caracter�sticas de los clientes que tienden o no a solicitar el dep�sito
# ofrecido por el banco, lo cual, ayudara sin duda a los planes comerciales de la empresa.


#------------------------------------------------------------------------------------------------------------
#                                                CONCLUSIONES
#------------------------------------------------------------------------------------------------------------

# El hecho de saber si un cliente estuvo afiliado al servicio de dep�sito a plazo fijo del banco en el pasado
# ayuda en gran medida a decidir si este volver� a solicitar nuevamente este servicio, ya que se evidenci� que
# aquellos clientes que s� estuvieron afiliados a este tipo de dep�sito en la campa�a anterior, volvieron a
# solicitarlo en la campa�a actual.

# Existe una tendencia en los clientes en solicitar pr�stamos de vivienda en mayo, lo que causa que en este mes
# haya menos posibilidades de conseguir que los clientes acepten solicitar este tipo de dep�sito. Es por ello
# que se puede recomendar al banco evitar invertir muchos esfuerzos y recursos en la persuasi�n de clientes en
# este mes del a�o, y centrar sus recursos en explotar otros meses de poca actividad como marzo, septiembre,
# octubre o diciembre, en donde se observ� una tendencia de los clientes a solicitar este dep�sito.

# Se evidencio que el tratamiento de outliers previo a la construcci�n de un modelo predictivo ayud� a que este
# obtuviera un rendimiento mucho mejor frente a modelos construidos con datos sin procesamiento previo de este
# tipo de valores. En donde la t�cnica utilizada fue la imputaci�n iterativa, aprovechando la potencia de los
# bosques aleatorios para reemplazar estos valores at�picos por valores que se asemejen a un comportamiento
# normal en nuestro conjunto de datos.

# Gracias a los valores SHAP, es posible dar una explicaci�n acerca del funcionamiento y la relevancia que tiene
# cada una de nuestras variables y los posibles valores que puedan tomar en la predicci�n de modelos de
# aprendizaje autom�tico de caja negra, aportando informaci�n muy �til que puede complementar en gran medida al
# an�lisis de datos que se haya realizado con anterioridad, o servir de antesala para nuevos an�lisis futuros.


