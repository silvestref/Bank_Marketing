#------------------------------------------------------------------------------------------------------------
#                                               INTRODUCCIÓN
#------------------------------------------------------------------------------------------------------------

# IDENTIFICACIÓN DEL PROBLEMA

# Uno de los usos más populares de la ciencia de datos es en el sector del marketing, puesto que es una
# herramienta muy poderosa que ayuda a las empresas a predecir de cierta forma el resultado de una campaña de
# marketing en base a experiencias pasadas, y que factores serán fundamentales para su éxito o fracaso. A la
# vez que también ayuda a conocer los perfiles de las personas que tienen más probabilidad de convertirse en
# futuros clientes con el fin de desarrollar estrategias personalizadas que puedan captar de forma más efectiva
# su interés. Conocer de antemano o a posteriori esta información es de vital importancia ya que ayuda en gran
# medida a que la empresa pueda conocer más acerca del público al que se tiene que enfocar, y que en el futuro
# se puedan desarrollar campañas de marketing que resulten más efectivas y eficientes. Entonces, se identifica
# que la problemática a tratar es el entender los factores que influyen a que una persona solicite o no un
# depósito a plazo fijo ofrecido por un determinado banco y predecir dado una serie de características, que
# personas solicitarán o no dicho servicio. Para ello, se requiere analizar la última campaña de marketing
# ejecutada por el banco y algunas características de sus clientes, con el fin de identificar patrones que nos
# puedan ayudar a comprender y encontrar soluciones para que el banco pueda desarrollar estrategias efectivas
# que les ayuden a captar el interés de las personas en solicitar este tipo de depósito, y en base a esto,
# construir un modelo predictivo que permita predecir que personas tomaran este servicio o no.


# ¿QUÉ ES UN DEPÓSITO A PLAZO FIJO?

# Es una inversión que consiste en el depósito de una cantidad determinada de dinero a una institución
# financiera por un periodo de tiempo, en donde el cliente no puede retirar el dinero depositado hasta que
# este periodo de tiempo haya finalizado. La ventaja de este tipo de depósito es que permite ahorrar dinero
# ganando intereses, por lo cual, muchas personas lo ven como una forma efectiva de generar ingresos pasivos.


# OBJETIVOS

# * Realizar análisis de datos para encontrar y entender los factores que influyen a que una persona solicite
#   o no un depósito a plazo fijo.
# * Construir un modelo de aprendizaje automático con CatBoost para la predicción de solicitantes de un depósito
#   a plazo fijo.
# * Implementar correctamente cada uno de los pasos de la metodología de ciencia de datos en la elaboración de
#   este proyecto


#------------------------------------------------------------------------------------------------------------
#                                   IMPORTACIÓN DE LIBRERÍAS Y CARGA DE DATOS
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
from matplotlib.ticker import FormatStrFormatter
import association_metrics as am
from collections import Counter
from catboost import CatBoostClassifier, Pool
import optuna  
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

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

# Podemos extraer algunos insights simples de esta tabla, como que el promedio de edad de los clientes de la
# empresa ronda en los 41 años. También que el saldo promedio que tienen en su cuenta es de 1528, pero si
# observamos la desviación estándar de los datos de esta variable, observamos que tiene un valor 3225, el cual
# es considerablemente alto, por lo que podemos decir que el saldo de los clientes está muy distribuido en
# nuestro conjunto de datos, presentando una alta variación. Por último, podemos observar que la variable pdays
# (número de días después del último contacto en la campaña anterior del banco) tiene un valor mínimo de -1,
# lo cual al momento de la interpretabilidad en el análisis de datos puede resultar algo confuso, es por ello
# que en la sección del preprocesamiento de datos se procederá a reemplazar este valor por un 0.

#------------------------------------------
#  ELIMINACIÓN Y CAMBIO DE TIPO DE VARIABLES
#------------------------------------------

# Hay que tener en cuenta algo de suma importancia en nuestros datos, y es que la variable "duration" hace
# referencia al tiempo de duración en segundos del último contacto que se realizó con la persona antes que
# decidiera solicitar o no un depósito a plazo fijo, y como naturalmente este valor no se conoce hasta después
# de haber realizado la llamada que es cuando ya se sabe la decisión de la persona, se procederá a eliminar al
# momento de construir nuestro modelo predictivo, puesto que estaría otorgando información que de por si no se
# conoce de antemano.

data.info()

# Observamos que aparentemente todas nuestras variables de entrada parecen tener cierta relación con la decisión
# de una persona en solicitar o no un depósito a plazo fijo, por lo que se decide por el momento no eliminar
# ninguna de estas variables de forma injustificada.

# También observamos que todas las variables de nuestro conjunto de datos están correctamente etiquetadas con
# el tipo de dato que les corresponde, por lo tanto, no se requiere realizar conversión alguna.


#------------------------------------------------------------------------------------------------------------
#                                           PREPROCESAMIENTO DE DATOS
#------------------------------------------------------------------------------------------------------------

# Como habíamos explicado en la sección anterior, procederemos a reemplazar los valores iguales a -1 por 0 en
# la variable pdays.

for i in range(0,data.shape[0]):
    if data["pdays"].iloc[i] == -1:
        data["pdays"].iloc[i] = 0
 
# Entonces, si ahora observamos el valor mínimo de la variable pdays obtendremos un 0 como resultado en vez de
# un -1.
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

# Con el diagrama de cajas observamos que tenemos presencia de outliers en todas nuestras variables numéricas
# excepto en la variable "day".

# A continuación, visualizaremos el porcentaje de outliers con respecto al total en cada una de nuestras
# variables para poder considerar si debemos tomar la decisión de eliminar alguna de estas variables por su
# alta presencia de valores atípicos.

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

( (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)) ).sum() / data.shape[0] * 100

# Los resultados nos arrojan que la variable "pdays" tiene un 24% de presencia de outliers respecto al total de
# filas, lo cual siguiendo la buena práctica de eliminar aquellas variables que superen un umbral del 15% de
# valores atípicos, procederemos a eliminar esta variable ya que puede inferir de forma negativa en el análisis
# y la predicción del futuro modelo de clasificación que se construirá. Otro dato a tomar en cuenta, es que
# esta variable es la misma que presentaba valores iguales a -1, los cuales reemplazamos con 0, donde quizá los
# valores etiquetados como -1 se debieron a una corrupción en los datos, con lo cual tenemos un motivo más
# para eliminar esta variable.

data = data.drop(["pdays"], axis=1)

# -- AGE --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["age"]], palette="Set3")
sns.distplot(data["age"], ax=ax[1])
plt.show()

# De los siguientes gráficos podemos observar que los datos que son catalogados como atípicos según el rango
# intercuartílico son personas que superan los 75 años de edad sin llegar a pasar los 95 años. Este rango de
# edad no es ningún error o corrupción en los datos, ya que la mayoría de personas con una calidad de vida
# adecuada podrían alcanzar este rango, por lo tanto, tenemos dos opciones para tratarlos:
    
# * Eliminar las filas que contengan estas edades debido a que su presencia es tan solo del 1.5% del total.
# * Imputarlos haciendo uso de un algoritmo predictivo.

# Todos estos métodos resultan aceptables, pero en este caso optaremos por imputarlos por un valor aproximado a
# lo "normal" que refleje la misma conducta que el valor atípico, más que todo para no perder información. De
# igual forma, al momento del entrenamiento y la elección del mejor modelo de clasificación, se comparará el
# rendimiento de un modelo libre de outliers con uno con outliers con el fin de observar si nuestra decisión
# fue acertada.


# -- CAMPAIGN --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["campaign"]], palette="Set3")
sns.distplot(data["campaign"], ax=ax[1])
plt.show()

# Con respecto a esta variable, observamos que la inmensa mayoría de nuestros datos tienen un valor entre 1 y 5,
# mientras que los datos atípicos adquieren valores superiores a este rango. Evidentemente este es un
# comportamiento inusual ya que, según nuestros datos, comúnmente solo se realizan entre 1 y 5 contactos con el
# cliente antes de que este tome una decisión final, por ende, números de contactos iguales a 10, 20, 30 e
# incluso mayores a 40 son demasiado extraños de ver. Por ende, procederemos a imputar estos valores por
# estimaciones que se aproximen a un valor común.

# -- PREVIOUS --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["previous"]], palette="Set3")
sns.distplot(data["previous"], ax=ax[1])
plt.show()

# Al igual que en la variable "campaign", "previous" aparte de tener una definición similar (número de contactos
# con el cliente en la campaña anterior), este también presenta un comportamiento similar, en donde se observa
# que los valores comunes están en un rango entre 0 y 3, y que los datos considerados como atípicos toman
# valores superiores a este rango, llegando incluso a ser excesivos. Es por ello que se tomara la misma
# decisión de imputarlos al igual que "campaign".

# -- BALANCE --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["balance"]], palette="Set3")
sns.distplot(data["balance"], ax=ax[1])
plt.show()

# Un comportamiento similar a las anteriores gráficas observamos en esta variable, donde nuevamente tenemos un
# sesgo por la derecha en donde los datos comunes adquieren valores entre -300 y 4000, y los que son atípicos
# llegan a superar fácilmente este umbral, aunque resulta más común que lo superen en forma positiva que en
# forma negativa, lo cual podemos deducir que, en términos de valores atípicos, es más común encontrar datos
# anormalmente altos que datos anormalmente bajos. Debido a que el porcentaje de datos atípicos para esta
# variable es del 9.4%, el cual no es un valor ni muy grande ni muy pequeño, no conviene eliminarlos, es por
# ello que los imputaremos por un nuevo valor aproximado que entre en un rango más común.

# -- DURATION --

fig, ax = plt.subplots(1, 2, figsize=(14,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data[["duration"]], palette="Set3")
sns.distplot(data["duration"], ax=ax[1])
plt.show()

# Esta variable también presenta un sesgo notorio por la derecha al igual que las variables anteriores, con la
# diferencia que su distribución parece ser más equitativa respecto a las demás, aquí podemos apreciar que los
# valores comunes están en un rango entre 0 y 1000 segundos (16 minutos aprox.) y que los que son considerados
# atípicos superan fácilmente este rango, llegando incluso a ser superiores a los 3000 segundos (50 minutos).
# Observar que una llamada entre un empleado del banco y un cliente supere los 30 minutos es un comportamiento
# inusual y que no se acostumbra a tener, es por ello que estos datos deben ser tratados, y para este caso
# haremos uso de la imputación iterativa aplicando bosques aleatorios para reemplazar dichos valores por unos
# que se acerquen a un comportamiento común de observar.


#----------------------------
# IMPUTACIÓN DE OUTLIERS
#----------------------------

# Crearemos una copia del conjunto de datos original con el fin de que mas adelante podamos comparar el
# rendimiento de nuestro modelo predictivo en ambos conjuntos (datos con outliers y sin outliers).

data2 = data.copy()

# El primer paso para realizar la imputación será convertir todos los valores atípicos que se hayan detectado
# mediante el rango intercuartílico por NaN, ya que la función que utilizaremos para la imputación trabaja con
# este tipo de datos.

outliers = (data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR))
data2[outliers] = np.nan

# Ahora tenemos que aplicar una codificación para nuestras variables categóricas, debido a que usaremos bosques
# aleatorios como medio de imputación, bastara con aplicar un label encoder.

# Nombres de nuestras variables categóricas
cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "deposit"]

# Diccionario para almacenar la codificación realizada en cada variable (Útil para después revertir la transformación)
dic = {}

for col in cols:
    dic[col] = LabelEncoder().fit(data2[col])
    data2[col] = dic[col].transform(data2[col])

# El siguiente paso ahora es dividir nuestros datos en conjuntos de entrenamiento y prueba con el fin de evitar
# la fuga de datos.

# Guardamos los nombres de las columnas de nuestro Dataset (Útil para después concatenar estos conjuntos en uno solo)
nom_cols = data2.columns.values

X = data2.iloc[: , :-1].values
y = data2.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)

# Finalmente, procederemos a realizar la imputación

imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=21), random_state=21)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Para visualizar el resultado de nuestra imputación de forma cómoda y gráfica, será necesario concatenar todos
# los subconjuntos que hemos creado en uno solo como teníamos inicialmente y revertir la codificación de
# nuestras variables categóricas.

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

data2 = pd.concat([X, y], axis=1)

data2.columns = nom_cols  # Se les introduce los nombres de las columnas con la variable anteriormente creada

# Se invierte la codificación
for col in cols:
    data2[col] = dic[col].inverse_transform(data2[col].astype(int))

# Debido a que las predicciones hechas por los bosques aleatorios se basan en el promedio del resultado de 
# varios arboles de decision, tendremos algunos datos imputados como decimal en variables que son enteras, como
# en el caso de "age", es por ello que redondearemos dichos valores decimales en cada variable que solo 
# contenga valores enteros

for col in ["age", "day", "campaign", "previous", "balance", "duration"]:
    data2[col] = data2[col].round()

# Ahora si podemos graficar para observar el cambio en nuestros datos después de la imputación.

fig, ax = plt.subplots(1, 3, figsize=(16,7))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(ax=ax[0], data= data2[["age", "day", "campaign", "previous"]], palette="Set3")
sns.boxplot(ax=ax[1], data= data2[["balance"]], palette="Pastel1")
sns.boxplot(ax=ax[2], data= data2[["duration"]], palette="Pastel1")
plt.show()

# Del grafico podemos observar que todas las variables a excepción de "balance" y "duration" están libres de
# outliers.

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data2[["balance"]], palette="Set3")
sns.distplot(data2["balance"], ax=ax[1])

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data2[["duration"]], palette="Set3")
sns.distplot(data2["duration"], ax=ax[1])

# Analizando las variables que aún tienen presencia de valores atípicos, se ve que la varianza en la distribución
# de estos valores ya no es tan extrema como teníamos inicialmente, si no que ahora se distribuyen en un rango
# menor a 1000 unidades, incluso pudiéndose acercar a una distribución normal.

Q1 = data2.quantile(0.25)
Q3 = data2.quantile(0.75)
IQR = Q3 - Q1

( (data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR)) ).sum() / data2.shape[0] * 100

# A la vez que también observamos que estos datos atípicos solo constituyen el 5.6% y 4.1% respectivamente del
# total, lo cual es una cifra moderadamente baja. Entonces podemos tomar dos decisiones, eliminarlos o
# conservarlos como parte de nuestros datos. En esta ocasión, elegiré conservarlos ya que pueden contener
# información útil para el análisis y para el modelo de clasificación, además que su presencia es relativamente
# baja con respecto del total y su distancia de los extremos no es tan alarmante ni exagerada.


#------------------------------------
# IDENTIFICACIÓN DE VALORES FALTANTES 
#------------------------------------

# Observamos cuantos valores faltantes hay en nuestro conjunto de datos
data2.isnull().sum().sum()

# Debido a que no hay presencia de valores faltantes o nulos, no será necesario tomar acciones al respecto



#------------------------------------------------------------------------------------------------------------
#                                      ANÁLISIS Y VISUALIZACIÓN DE DATOS
#------------------------------------------------------------------------------------------------------------

# En base a las variables que tenemos disponible empezaremos la sección formulando algunas hipótesis que
# seran respondidas mediante el proceso de análisis de los datos.

# H1: ¿Es la edad del cliente un factor que propicie la solicitud de un deposito a plazo fijo?
# H2: ¿Que tipo de trabajos son mas propensos a tener clientes que quieran solicitar un deposito a plazo fijo?
# H3: ¿Los clientes casados son menos propensos a solicitar un deposito a plazo fijo?
# H4: ¿El grado de educacion alcanzado por el cliente propicia a la solicitud de un deposito a plazo fijo?
# H5: ¿Los clientes con mora crediticia en el banco son menos propensos a solicitar un deposito a plazo fijo?
# H6: ¿Se puede decir que los clientes con mayor dinero en su cuenta bancaria son muy propensos a solicitar un
# deposito a plazo fijo?
# H7: ¿Los clientes con un prestamo para vivienda en el banco son menos propensos a solicitar un deposito a plazo fijo?
# H8: ¿Los clientes con un prestamo personal en el banco son menos propensos a solicitar un deposito a plazo fijo?
# H9: ¿El medio de comunicacion con el que se contacta con el cliente afecta en la solicitud de un deposito a plazo fijo?
# H10: ¿Existen dias especificos en los que sea mas probable convencer a un cliente de solicitar un deposito a plazo fijo?
# H11: ¿Existen meses especificos en los que sea mas probable convencer a un cliente de solicitar un deposito a plazo fijo?
# H12: ¿Se puede decir que a mayor duracion en tiempo de constacto con el cliente aumentan las posibilidades de
# que este acepte solicitar un deposito a plazo fijo?
# H13: ¿Es cierto que mientras mas contactos se tenga con el cliente mayor sera la posibilidad de que este
# termine aceptando solicitar un deposito a plazo fijo?
# H14: ¿El numero de contactos realizados en la campaña anterior afecta en la posibilidad de que los clientes
# soliciten un deposito a plazo fijo?
# H15: ¿Los clientes que solicitaron un deposito a plazo fijo en la campaña anterior son mas propensos a solicitar
# el mismo servicio en la campaña actual?


#--------------------
# ANALISIS UNIVARIADO
#--------------------

# Para comenzar, visualizaremos la distribución de los datos respecto a cada uno de los tres conjuntos de
# variables que se han identificado: Variables de información del cliente - Variables de informacion bancaria
# - Variables de campaña. Esta segmentación nos permitirá realizar un análisis mas ordenado e identificar
# patrones e información util para entender nuestros datos.


# VARIABLES DE INFORMACIÓN DEL CLIENTE

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.histplot(data=data2, x="age", kde=True, ax=ax[0,0], color="g")
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
fig.suptitle('Distribución de las variables de información del cliente', fontsize=16)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
sns.countplot(data=data2, y="job")
ax.set_title("job")
ax.set_ylabel("")
plt.show()

# Observamos que la mayoria de clientes del banco tienen edades que entran en el rango de los 30 y 40 años, sin
# embargo, la diferencia entre el numero de clientes que entran en este rango y los que no, no es muy grande.
# Entonces podemos decir que el banco en su gran mayoria tiene clientes que no sobrepasan la mediana edad.

# Tambien podemos observar que la mayoria de estas personas son casadas y que muy pocas son divorciadas. A la
# vez que tambien se ve que el tipo de educacion predominante es la secundaria y terciaria, lo cual tiene 
# sentido ya que en estos niveles de eduacion se aprenden materias que guardan relacion con la economia y la
# sociedad.

# Se aprecia tambien que el medio de contacto preferido por los clientes es el celular

# Por ultimo, la mayoria de clientes del banco tienen puestos de gerencia, obrero y tecnico, y muy pocos son
# amas de casa, emprendedores o desempleados.


# VARIABLES DE INFORMACIÓN BANCARIA

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
fig.suptitle('Distribución de las variables de información del cliente', fontsize=16)
plt.show()

# Con respecto a la variable "balance" (saldo del cliente en su cuenta bancaria) observamos que existen muchos
# clientes que tienen relativamente poco dinero acumulado en sus cuentas, estos valores se encuentran en un
# rango mayor a 0 y menor a 1000.

# Tambien podemos observar que casi no existen clientes morosos en el banco, esta variable se podria relacionar
# con "balance" en donde se observa que hay muy pocas personas con saldo negativo en sus cuentas bancarias

# Por otro lado, tenemos que la cantidad de clientes que han solicitado un prestamo para vivienda es muy
# similar a la cantidad de clientes que no solicitaron dicho prestamo.

# Por ultimo, observamos que la gran mayoria de clientes no han solicitado un prestamo personal, y los que si
# lo han hecho, debido a que son minoria, podrian relacionarse con la poca presencia de clientes deudores en 
# la variable "default" y la poca presencia de clientes con saldo negativo en la variable "balance".


# VARIABLES DE CAMPAÑA
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="day", ax=ax[0])
ax[0].set_title("day")
ax[0].set_xlabel("")
ax[0].set_xticklabels(range(1,32))
sns.countplot(data=data2, x="month", ax=ax[1], order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                                                      'sep', 'oct', "nov", "dec"])
ax[1].set_title("month")
ax[1].set_xlabel("")
fig.suptitle('Distribución de las variables de información del cliente', fontsize=16)
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.histplot(data=data2, x="duration", kde=True, ax=ax[0,0], color="g")
ax[0,0].set_title("duration")
ax[0,0].set_xlabel("")
sns.countplot(data=data2, x="campaign", ax=ax[0,1])
ax[0,1].set_title("campaign")
ax[0,1].set_xlabel("")
sns.countplot(data=data2, x="previous", ax=ax[1,0])
ax[1,0].set_title("previous")
ax[1,0].set_xlabel("")
sns.countplot(data=data2, x="poutcome", ax=ax[1,1])
ax[1,1].set_title("poutcome")
ax[1,1].set_xlabel("")
fig.suptitle('Distribución de las variables de información del cliente', fontsize=16)
plt.show()

# Observamos que la cantidad de veces con respecto a los dias en los que se contacta al cliente por ultima vez
# estan distribuidos de forma casi equitativa, en donde solo se observan picos muy bajos en los dias 1, 10, 24
# y 31 de cada mes, y los picos mas altos son los que se acercan a principio, quincena o final de cada mes. 
# Esto se debe probablemente a que estos dias son previos al pago que reciben los clientes en su trabajo, por
# ende, se intenta aprovechar el ingreso de dinero que tienen esos dias para ofrecerles los servicios de deposito
# a plazo fijo.

# Con respecto al mes en el que se realizo el ultimo contacto, se observa que Mayo es un mes con mucha importancia
# para realizar contacto con los clientes, esto podria deberse a que los datos que estamos analizando provienen
# de un banco en Portugal, y dicho pais celebra en este mes el dia del trabajo, por lo tanto se puede suponer
# que todos los trabajadores en dicho pais reciben un incentivo economico y el banco aprovecha esta situacion
# para que el dinero recibido por sus clientes se invierta en la empresa.

# Por otro lado, respecto a la duracion del ultimo contacto en segundos, podemos ver que la gran mayoria de estos
# tuvo una duracion entorno a los 100 y 300 segundos (1.6 y 3.3 minutos respectivamente), lo cual es un tiempo
# justo para saber la decision final del cliente

# El numero de contactos realizados en esta campaña son en su gran mayoria 1 o 2. Y el numero de contactos 
# realizados en la anterior campaña esta muy inclinado al 0, por lo tanto se puede deducir que el banco tiene
# nuevos clientes o que la campaña anterior no fue ejecutada de forma adecuada.

# Por ultimo, respecto a la variable "poutcome" (resultado de la campaña anterior) podemos observar que una
# inmensa mayoria de clientes estan etiquetados como "unknown" (desconocido), lo cual respalda la suposicion
# que anteriormente habiamos hecho respecto a que el banco tenia nuevo clientes, debido a que esta variable 
# guarda relacion con "previous".


# Una vez conocida la distribución de las variables con las que vamos a trabajar, procederemos a responder las
# hipótesis que inicialmente habíamos planteado, esto lo lograremos mediante un análisis bivariado de nuestras
# variables de entrada con nuestra variable de salida.


#-------------------
# ANALISIS BIVARIADO
#-------------------

# VARIABLES DE INFORMACIÓN DEL CLIENTE VS "deposit"

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(hspace=0.3)
sns.histplot(data=data2, x="age", kde=True, ax=ax[0,0], hue=data2.deposit, multiple="stack", color="g")
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
fig.suptitle('Variables de información del cliente vs Churn', fontsize=16)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
sns.countplot(data=data2, x="job", hue=data2.deposit)
ax.set_title("job")
ax.set_ylabel("")
plt.show()

# En primer lugar, mediante el histograma observamos que la curva de densidad de las edades de los clientes que
# solicitarion y no solicitaron el deposito son muy similares, obteniendo en ambos casos los picos mas altos
# en edades que entran en el rango de los 30 y 40 años, y que estos picos se diferencian por relativamente
# pocas cifras de las demas edades. Es decir, no existe un patron claro que indique de forma significativa que
# una edad en especifico es mas propensa a solicitar un deposito a plazo fijo o no.

# Por otro lado, podemos observar que la variable "marital" no presenta relacion alguna con la solicitud de un
# deposito a plazo fijo, ya que la cantidad de clientes que solicitaron o no el deposito se reparten de forma
# equitativa entre los que son solteros, casados y divorciados

# El mismo comportamiento se puede apreciar en la variable "education", donde la cantidad de clientes solicitantes
# y no solicitantes son muy parecidas en todos los grados de educacion.

# Con respecto a "contact", podemos identificar que los clientes con un medio de comunicacion desconocido por
# el banco son menos propensos a solicitar un deposito a plazo fijo, esta informacion podria no ser tan
# relevante debido a que como el medio de comunicacion es desconocido, estos datos podrian ir a cualquier de
# las dos categorias restantes, sesgando un poco el resultado del analisis.

# Por ultimo, en la variable "job" podemos observar que los clientes con trabajo "blue-collar" (obrero) son
# menos propensos a solicitar un deposito a plazo fijo, pobrablemente por los pocos ingresos que se obtienen
# de esta labor, por otra parte, observamos que los estudiantes (student) y los retirados (retired) son 
# levemente mas propensos a solicitar este tipo de deposito, posiblemente debido a la cultura financiera que 
# existe en la mayoria de centros educativos y la alta disponibilidad de dinero que se tiene al haberse jubilado.

# Resumiendo toda la informacion obtenida tenemos que: La edad de los clientes no es un factor muy influyente
# para determinar si estos van a solicitar un deposito a plazo fijo o no, ademas que tanto su estado marital
# como educacional tampoco influyen en esta decision, sin embargo, se observa que los clientes con un medio de
# contacto desconocido por el banco son mas propensos a no solicitar este tipo de deposito, a la vez que los
# que tienen trabajos relacionados con la mano de obra tienen una tendencia a tampoco solicitar este servicio,
# y las personas que son estudiantes o retiradas a menudo aceptan el deposito a plazo fijo.

# Respondiendo a las hipótesis tenemos que:
# H1: La edad del cliente no afecta de forma significativa en la decision de solicitar un deposito a plazo fijo.
# H2: Se observo que los estudiantes y las personas retiradas son ligeramente mas propensas a solicitar un
# deposito a plazo fijo
# H3: El estado marital del cliente no influye en la decision de solicitar un deposito a plazo fijo
# H4: El grado de educacion alcanzado por el cliente no inluye de forma significativa en la decision de
# solicitar un deposito a plazo fijo
# H9: Los clientes con un medio de contacto desconocido por el banco tienen ligeramente mas probabilidad de no
# solicitar un deposito a plazo fijo


# VARIABLES DE INFORMACIÓN BANCARIA VS "deposit"

fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(hspace=0.3)
sns.histplot(data=data2, x="balance", kde=True, ax=ax[0,0], hue=data2.deposit, multiple="stack", color="g")
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
fig.suptitle('Variables de información del cliente vs Churn', fontsize=16)
plt.show()

# Del primer grafico, mediante el histograma observamos que las curvas de densidad del saldo de los clientes en
# el banco son muy similares para los que solicitaron y no solicitaron el deposito a plazo fijo, siguiendo
# casi una distribucion normal, no se observa algun patron marcado que indique que un rango de saldo en especifico
# propicie o no la solicitud de un deposito a plazo fijo.

# Con respecto a "default", tampoco se puede observar algun patron especifico que indique que el ser un cliente
# moroso o no afecte en la decision de solicitar o no un deposito a plazo fijo, ya que la distribucion de estos
# se reparten de forma equitativa en ambas ocaciones

# Con "housing" no podemos decir lo mismo, ya que aqui si se aprecia que los clientes que no solicitaron un
# prestamo de vivienda tienen una tendencia a solicitar un deposito a plazo fijo, mientras que los que si
# solicitaron un prestamo de vivienda tienen una tendencia a no solicitar este tipo de deposito. Esto podria
# deberse a que como ya tienen una deuda con el banco, ese dinero solicitado no puede destinarse a otros fines
# que no sea la adquision de una propiedad.

# Por ultimo, en la variable "loan" nuevamente no se observa un patron claro que indique una inclinacion hacia
# solicitar o no solicitar un deposito a plazo fijo si el cliente ha solicitado un prestamo personal o no.

# Resumiendo toda la informacion obtenida tenemos que: El saldo de los clientes en sus cuentas bancarias no
# es un factor del que se pueda deducir si estos en un futuro solicitaran un deposito a plazo fijo o no, lo 
# mismo podemos decir con respecto a si este cliente tiene mora crediticia o no, y si este solicito un prestamo
# personal o no. Sin embargo, observamos un patron claro que indica que los clientes que solicitaron un prestamo
# de vivienda son menos propensos a solicitar un deposito a plazo fijo, probablemente porque ese dinero solicitado
# sera destinado a otros fines.

# Respondiendo a las hipótesis tenemos que:
# H5: El hecho de tener o no tener mora crediticia no influye en la decision de solicitar o no un deposito a
# plazo fijo
# H6: El dinero que los clientes tengan en su cuenta bancaria no influye en la decision de solicitar o no un
# deposito a plazo fijo
# H7: Los clientes que solicitaron un prestamo de vivienda al banco son menos propensos a solicitar un deposito
# a plazo fijo
# H8: El hecho de solicitar o no un prestamo personal al banco no influye de forma significativa en la
# decision de solicitar o no un deposito a plazo fijo


# VARIABLES DE CAMPAÑA VS "deposit"

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="day", hue=data2.deposit, ax=ax[0])
ax[0].set_title("day")
ax[0].set_xlabel("")
ax[0].set_xticklabels(range(1,32))
sns.countplot(data=data2, x="month", ax=ax[1], order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                                                      'sep', 'oct', "nov", "dec"], hue=data2.deposit)
ax[1].set_title("month")
ax[1].set_xlabel("")
fig.suptitle('Distribución de las variables de información del cliente', fontsize=16)
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.histplot(data=data2, x="duration", kde=True, hue=data2.deposit, ax=ax[0,0], multiple="stack", color="g")
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
fig.suptitle('Distribución de las variables de información del cliente', fontsize=16)
plt.show()

# En primer lugar podemos observar que los dias en donde se tuvo mayor exito con respecto a la solicitud de un
# deposito a plazo fijo son los dias 1 y 10 de cada mes, el exito que se tiene en el dia 1 puede deberse a que
# este es un dia despues del que los clientes reciben su pago mensual por laborar, entonces, al tener una
# cantidad considerable en sus manos, es mas facil presuadirlos para que lo inviertan en el banco, tambien
# podemos ver que este dia es uno de los que menos contacto se tiene con el cliente, por lo tanto se podria
# recomendar para la proxima campaña aprovechar este dia para persuadir a mas personas. Con respecto a los dias
# en los que no se tuvo exito, podemos observar que estos son el 19, 20, 28 y 29 de cada mes.

# Por otra parte, observamos que los meses en los que se tuvo mayor exito fueron marzo, septiembre, octubre y
# en menor medida diciembre, mientras que el mes en el que se tuvo un mayor fracaso fue mayo.

# Con respecto a la variable "duration", se puede observar que en un principio los pocos segundos de comunicacion
# con el cliente tienen una proporcion similar de clientes que solicitarion y no solicitaron el deposito, y que
# a medida que el tiempo de contacto se vaya prolongando, hay mejores probabilidades de que este termine 
# aceptando realizar este tipo de deposito. Esta es una conducta normal, ya que cuando una persona esta
# interesada en adquirir algun producto o servicio, surgen diversas preguntas acerca de ello, lo cual, naturalmente
# prolonga el tiempo de comunicacion con el individuo que brinda dicho servicio.

# En la variable "campaign" se puede apreciar que no existe un patron que indique con certeza que un determinado
# numero de contactos favorece a la solicitud de un deposito a plazo fijo, aunque observamos que existen mas
# clientes que aceptaron realizar el deposito cuando se realizo solo 1 contacto con ellos, la diferencia entre
# los que aceptaron o no, no es muy grande para considerarlo relevante.

# Con "previous" no podemos decir lo mismo, ya que se ve que los clientes que no han sido contactados en una
# campaña anterior para ofrecerles este tipo de deposito son menos propensos a aceptar dicho deposito en la
# campaña actual, mientras que aquellos que si han sido contactados anteriormente, tienen una leve inclinacion
# a solicitar este tipo de servicio.

# Por ultimo, respecto a la variable "poutcome", podemos observar que aquellos clientes de los que no se sabe
# si aceptaron o no solicitar un deposito a plazo fijo tienen una tendencia a no aceptar este tipo de deposito
# en la campaña actual, cabe mencionar que si su decision fue etiquetada como desconocida se podria deberse a que
# son nuevos clientes, ya que esta variable guarda relacion con "previous", en donde se puede observar
# que la cantidad de clientes a los que no se les han contactado en la campaña anterior (0) es la misma que los
# que los que estan etiquetados como "unknown". Por otro lado observamos que aquellos clientes a los cuales se
# les pudo persuadir para solicitar este tipo de deposito en la campaña anterior, con mucha probabilidad volveran
# a aceptar solicitar este servicio en la campaña actual.

# Resumiendo toda la informacion obtenida tenemos que: Los dias que registraron mayor exito en la solicitud de
# un deposito a plazo fijo fueron los 1 y 10 de cada mes, mientras que los que menos exito tuvieron fueron los
# dias 19, 20, 28 y 29. Asimismo los meses de mayor exito fueron Marzo, Septiembre, Octubre y Diciembre, y el 
# de menor exito Mayo. Tambien se observo que si se tiene una comunicacion corta con el cliente, la posibilidad
# que este acepte solicitar este tipo de deposito es casi la misma que la de no solicitarlo, y que mientras
# mayor sea el tiempo de contacto, mayor sera la posibilidad de tener exito en su persuacion. El numero de
# contactos que se tiene con el cliente parece no afectar en su decision, sin embargo, variables referentes a
# la campaña anterior como "previous" y "poutcome" parecen si afectar en esta decision, donde se pudo identificar
# que aquellos clientes a los cuales no se les contacto en la campaña anterior y cuyo resultado de si aceptaron
# solicitar el deposito a plazo fijo o no es desconocido, tienen una tendencia a no solicitar este tipo de 
# deposito en la campaña actual, mientras que aquellos de los que se sabe que si aceptaron solicitar este
# deposito en la campaña anterior, con mucha probabilidad volveran a solicitarlo en la campaña actual.

# Respondiendo a las hipótesis tenemos que:
# H10: Los dias en los que se observo que es mas probable convencer a los clientes de solicitar un deposito a
# plazo fijo fueron el 1 y 10 de cada mes
# H11: Los meses en los que se observo que es mas probable convencer a los clientes de solicitar un deposito a
# plazo fijo fueron marzo, septiembre, octubre y diciembre
# H12: A mayor duracion en el tiempo de contacto con el cliente mayores posibilidades hay de que este termine
# aceptando solicitar un deposito a plazo fijo
# H13: El numero de contactos que se tiene con el cliente parece no afectar en su decision de solicitar o no
# un deposito a plazo fijo
# H14: El numero de contactos que se tuvo con el cliente en la campaña anterior afecta en la posibilidad de 
# solicitar un deposito a plazo fijo
# H15: Aquellos clientes que solicitaron un deposito a plazo fijo en la campaña anterior con mucha probabilidad
# volveran a solicitar este tipo de deposito en la campaña actual


# A lo largo del proceso de analisis para responder las hipotesis que inicialmente habiamos planteado, nos
# hemos encontrado con algunos comportamientos y patrones particulares de los cuales se puede extraer informacion 
# relevante para el analisis. Es por ello que en esta parte se compararan entre si algunas de las variables mas
# relevantes con respecto a la decision de solicitar un deposito a plazo fijo con el fin de obtener insights 
# que nos ayuden a entender un poco mas el comportamiento de los clientes.


# "job" vs "housing"

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="job", hue=data2.housing)
ax.set_title("job")
ax.set_xlabel("")
plt.show()

# Del grafico podemos observar que los clientes que trabajan de obrero son los que en su gran mayoria solicitan
# un prestamo de vivienda, es por ello que las personas con este tipo de trabajo son menos propensas a solicitar
# un deposito a plazo fijo, ya que como vimos en analisis anteriores, el dinero que piden prestado al banco
# va destinado a otros fines que no son los buscados en este analisis. Por otra parte, podemos observar que las
# personas que son estudiantes o retirados son menos propensas a solicitar un prestamo de vivienda, por lo
# tanto, uniendo los hilos con el analisis respecto a la variable "housing", es de esperar que estas personas
# tengan mas probabilidades de solicitar un deposito a plazo fijo puesto que no tienen deudas con el banco, y 
# probablemente su cultura financiera o experiencia les hace mas atractivo el hecho de invertir que de gastar.


# "previous" vs "poutcome"

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="previous", hue=data2.poutcome)
ax.set_title("previous")
ax.set_xlabel("")
plt.show()

# Del siguiente grafico obtenemos un patron muy obvio en donde los clientes que no han sido contactados en la
# campaña anterior, estan etiquetados como resultado desconocido en si solicitaron o no un deposito a plazo 
# fijo en la campaña anterior. Por otra parte, podemos observar que efectivamente el numero de contactos que
# se tiene con el cliente no afecta en su decision de solicitar o no este tipo de deposito, ya que como se
# puede apreciar, las personas que solicitaron y no solicitar el deposito, se distribuyen de forma muy
# equitativa.


# "job" vs "duration"

duration_mean = data2.groupby(["job"], as_index=False)["duration"].mean()

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.barplot(data=duration_mean, x="duration", y="job")
ax.set_title("previous")
ax.set_xlabel("")
plt.show()

# Por ultimo, podemos observar que la media de tiempo de contacto que se tiene con cada uno de los clientes
# pertenecientes a los distintos tipos de trabajo se distribuye de forma muy equitativa, donde la diferencia 
# maxima que se puede apreciar es de 1 minuto. Aunque podemos ver que el tiempo de contacto que se tienen con
# los clientes que se encuentran desempleados es ligeramente mayor al resto, esto podria deberse a que la 
# situacion de estas personas les obligan a tener una fuente de ingresos para poder subsistir, por lo tanto,
# el tiempo de contacto con ellos se ve mas prolongado al tener mas interes en consultar como es el
# funcionamiento de este tipo de deposito y sus beneficios.


# Para terminar con esta sección, graficaremos una matriz de correlación para identificar el comportamiento
# conjunto de nuestras variables sobre otras. Como estamos tratando tanto con variables categóricas como
# numéricas, será necesario aplicar la correlacion de Pearson para las caracteristicas numericas, y la V de
# Cramer para las categoricas.


#-----------------------
# CORRELACION DE PEARSON
#-----------------------

data_corr = data2.copy()

data_corr["deposit"] = LabelEncoder().fit_transform(data_corr["deposit"])

plt.figure(figsize=(30, 20))
corr = data_corr[["age","balance","day","duration","campaign","previous","deposit"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)

# De la matriz observamos que las variables numericas con mayor correlacion hacia nuestra variable dependiente
# son "duration" y "previous". La influencia de estas variables ya lo habiamos analizado y gracias a esta matriz
# nuestras suposiciones estan mejor respaldadas. Respecto a "duration", habiamos llegado a la conclusion que
# mientras mayor era el numero de segundos en el que se mantenia contacto con el cliente, mayores eran las
# posibilidades de que este terminara aceptando solicitar el deposito. Y con respecto a "previous", identificamos
# que los clientes que no habian sido contactados en una campaña anterior, tenian mas probabilidad de no 
# solicitar el deposito en la campaña actual


#------------
# V DE CRAMER
#------------

data_corr = data2.copy()

data_corr = data_corr.apply(lambda x: x.astype("category") if x.dtype == "O" else x)
cramersv = am.CramersV(data_corr) 
result = cramersv.fit()

# Con respecto a la asociacion entre nuestras variables categoricas y nuestra variable dependiente podemos 
# observar que aquellas cuyo valor de asociacion es mayor que el resto son "housing", "contact", "month" y
# "poutcome", las cuales habiamos en analisis anteriores que presentaban ciertos patrones que indicaban la
# inclinacion del cliente hacia solicitar o no solicitar un deposito a plazo fijo. Donde los clientes que
# solicitaron un prestamo de vivienda eran menos propensos a solicitar este tipo de deposito, al iguales que
# los clientes de los que no se conocia el medio de comunicacion por el cual se les contactaba. Con respecto
# a los meses observamos que habian algunos en los que se tenian resultados muchos mas positivos y otros en
# los que no habia mucho exito. Por ultimo, tambien pudimos identificar que aquellos clientes que habian 
# solicitado realizar este tipo de deposito en la campaña anterior con mucha probabilidad volverian a solicitarlo
# en la campaña actual, mientras que aquellos que eran nuevos en el banco y no se tenia un registro acerca de
# su decision tenian una tendencia a no solicitar este servicio.

# Gracias al valor de la asociacion de Cramer tambien podemos obtener algunos otros insights interesantes como:
    
# "month" vs "housing"

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="month", order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
                                            'oct', "nov", "dec"], hue=data2.housing)

# En donde podemos observar que existe un numero significativo de clientes que han sido contactados por ultima
# vez en Mayo y que han solicitado un prestamo de vivienda. Esto podria indicar que los clientes tienen una
# tendencia a solicitar este prestamo un mes antes de Mayo, ya que se puede observar como en Abril el numero
# de personas con esta solicitud van en aumento, y como pasado el mes de Mayo este numero decrece, volviendo
# a un estado estandar en el que el numero de personas que no solicitaron este tipo de prestamos son mayores o
# iguales a las que si lo solicitaron.


# "job" vs "education"

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
sns.countplot(data=data2, x="job", hue=data2.education)
ax.set_title("job")
ax.set_xlabel("")

# Por ultimo, podemos observar un patron completamente normal en donde la mayoria de personas que tienen cargos
# relacionados con la gerencia, tienen estudios terciarios (universitarios o de instituto). Y que los demas
# puestos de trabajo estan conformados por personas cuyo grado de educacion mayormente es secundario, excepto
# en el caso de los obreros, retirados y amas de casa, donde la distribucion entre las personas con educacion
# secundaria y primaria es casi equitativa.


#------------------------------------------------------------------------------------------------------------
#                                           TRANSFORMACIÓN DE DATOS
#------------------------------------------------------------------------------------------------------------

# Empezaremos por eliminar la variable "duration" de la cual anteriormente habiamos hablado, ya que aporta
# informacion de la cual no se dispone en la realidad al momento de predecir si un cliente solicitara o no un
# deposito a plazo fijo, ya que la duracion de la llamada con el cliente se conoce despues de saber la decision
# de este, mas no antes.

data = data.drop(["duration"], axis=1)
data2 = data2.drop(["duration"], axis=1)

#---------------------------
# CODIFICIACION DE VARIABLES
#---------------------------

# Como uno de los objetivos de este proyecto es implementar CatBoost para la prediccion de clientes que
# solicitaran o no un deposito a plazo fijo en el futuro, no sera necesario codificar de forma manual nuestras
# variables categoricas, ya que CatBoost internamente realiza este proceso por nosotros, implementando una
# codicaficacion basada en Target Encoder con algunas modificaciones que el algoritmo cree pertinente. Solo
# seria necesario aplicar una codificacion de etiqueta si nuestra variable dependiente es dicotomica. Sin
# embargo, para demostrar que efectividad tiene el delegarle la codificacion a CatBoost y hacerlo de forma
# manual en la precision de nuestro modelo, construiremos dos modelos utilizando ambas tecnicas y posteriomente
# evaluaremos su rendimiento


# CON CODIFICACION MANUAL
#------------------------

# Puesto que el algoritmo que vamos a utilizar esta basado en árboles de decision, para evitar el aumento
# exponencial de variables independientes al implementar una codificacion One Hot Econding y todos los problemas
# que esto conyeva, podemos utilizar Label Encoder como alternativa, ya que los arboles de decision no se ven
# perjudicados al tener variables ordinales que originalmente son nominales

# Codificacion de variables en el conjunto con outliers
data_cod = data.copy()

cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "deposit"]

for col in cols:
    data_cod[col] = LabelEncoder().fit_transform(data_cod[col])
    
# Codificacion de variables en el conjunto sin outliers

data2_cod = data2.copy()

for col in cols:
    data2_cod[col] = LabelEncoder().fit_transform(data2_cod[col])


# SIN CODIFICACION MANUAL
#------------------------

# Codificacion de etiqueta a la variable dependiente del conjunto con outliers 
data["deposit"] = LabelEncoder().fit_transform(data["deposit"])

# Codificacion de etiqueta a la variable dependiente del conjunto sin outliers 
data2["deposit"] = LabelEncoder().fit_transform(data2["deposit"])


#----------------------------------------------------
# CREACIÓN DE CONJUNTOS DE ENTRENAMIENTO Y VALIDACIÓN
#----------------------------------------------------

# PARA DATOS CON OUTLIERS Y SIN CODIFICACION MANUAL
#--------------------------------------------------
X = data.iloc[: , :-1].values
y = data.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21, stratify=y)


# PARA DATOS CON OUTLIERS Y CON CODIFICACION MANUAL
#--------------------------------------------------
X_cod = data_cod.iloc[: , :-1].values
y_cod = data_cod.iloc[: , -1].values

X_train_cod, X_test_cod, y_train_cod, y_test_cod = train_test_split(X_cod, y_cod, test_size=0.30,
                                                                    random_state=21, stratify=y)


# PARA DATOS SIN OUTLIERS Y SIN CODIFICACION MANUAL
#--------------------------------------------------
X2 = data2.iloc[: , :-1].values
y2 = data2.iloc[: , -1].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.30, random_state=21, stratify=y)


# PARA DATOS SIN OUTLIERS Y CON CODIFICACION MANUAL
#--------------------------------------------------
X2_cod = data2_cod.iloc[: , :-1].values
y2_cod = data2_cod.iloc[: , -1].values

X2_train_cod, X2_test_cod, y2_train_cod, y2_test_cod = train_test_split(X2_cod, y2_cod, test_size=0.30,
                                                                        random_state=21, stratify=y)


#--------------------
# REBALANCEO DE DATOS
#--------------------

# Empezaremos comprobando el número de muestras para cada una de las clases que tiene nuestra variable dependiente
# para identificar si tenemos un conjunto de datos desbalanceado.

plt.figure(figsize=(15, 8))
sns.countplot(data=data2, x="deposit", palette=["#66c2a5", "#fc8d62"])
plt.title("Distribución del número de muestras", fontsize=20)
plt.show()

counter_total = Counter(data["deposit"])
print(counter_total)

# Observamos que no tenemos una desproporcion muy grave con respecto al numero de muestras en cada clase, por
# lo tanto, podemos obviar el uso de tecnicas de sobremuestreo y submuestreo para el rebalanceo de muestras.
# Cabe mencionar que CatBoost tambien posee un hiperparámetro encargado de solucionar este problema, añadiendo
# pesos a las muestras de la clase minoritaria para que su impacto en el modelo sea casi el mismo que el de la
# clase mayoritaria, por lo tanto, podriamos hacer uso de esta funcion para mejorar un poco mas el rendimiento
# predictivo de nuestro modelo.


#------------------------------------------------------------------------------------------------------------
#                               CONSTRUCCIÓN Y EVALUACIÓN DEL MODELO PREDICTIVO
#------------------------------------------------------------------------------------------------------------

# Como ya se menciono en la introducción de este proyecto, para la construccion de un modelo predictivo
# utilizaremos CatBoost.

# El motivo principal por el que elegimos este algoritmo basado en el aumento del gradiente es porque
# ofrece soporte para el trabajo de clasificacion y regresion con variables categoricas, ademas que en la
# mayoria de ocaciones se puede obtener resultados considerablemente buenos sin realizar demasiados ajustes
# en los hiperparametros, y por ultimo, porque es relativamente rapido entrenarlo, incluso cuando se tiene una
# cantidad considerable de datos. Estas cualidades encajan bien con nuestro conjunto de datos, puesto que 
# tenemos alrededor de 11000 observaciones las cuales tienen caracteristicas pertenecientes tanto a variables
# categoricas como numericas.


#-----------------------------
# ELECCIÓN DE HIPERPARÁMETROS
#-----------------------------

# Como anteriormente habiamos dicho, CatBoost puede obtener resultados buenos con la configuracion de
# hiperparametros predeterminada, sin embargo, el objetivo de este proyecto es obtener el mejor modelo posible
# que pueda predecir de forma correcta la solicitud de deposito a plazo fijo de los clientes, es por ello que
# haciendo uso de la librería Optuna, intentaremos encontrar la combinación de hiperparametros que mejor se
# ajuste a nuestros datos.

# Dado que a lo largo de este proyecto hemos realizado distintas transformaciones a nuestros datos, y hemos
# guardado una copia del conjunto de datos antes de realizar dicha transformacion, aplicaremos la funcion de
# busqueda de hiperparametros a cada uno de estos conjuntos, con el fin de comparar hasta que paso de la transformación
# es necesaria para obtener el modelo con el mejor rendimiento posible, o si para este caso, no es necesario
# aplicar transformacion alguna. Es por ello que dividiremos esta seccion en cuatro partes, basado en los 
# cuatro conjuntos de datos obtenidos:
    
# Hiperparámetros para datos con outliers y sin codificacion manual
# Hiperparámetros para datos con outliers y con codificacion manual
# Hiperparámetros para datos sin outliers y sin codificacion manual
# Hiperparámetros para datos sin outliers y con codificacion manual

#------------------------------
# HIPERPARÁMETROS PARA DATOS CON OUTLIERS Y SIN CODIFICACION MANUAL

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
    
    # Identificación de variables categoricas
    categorical_features_indices = np.where(data.dtypes == np.object)[0]
    
    train_pool = Pool(X_train, y_train, cat_features = categorical_features_indices)
    test_pool = Pool(X_test, y_test, cat_features = categorical_features_indices)
    
    # Inicialización y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluación y obtención de métricas
    preds = model.predict(X_test)
    metric = accuracy_score(y_test, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_1 = study.trials_dataframe()

# Se ejecutó la función tres veces de forma independiente, y posterior a ello, se registro
# la mejor combinación de parámetros que arrojo cada ejecución, siendo estas las siguientes:

# 73.96% | iterarions=600, learning_rate=0.166129, depth=7, l2_leaf_reg=0.963535, random_strength=11, bagging_temperature=1, max_ctr_complexity=0
# 74.08% | iterarions=1200, learning_rate=0.246729, depth=9, l2_leaf_reg=7.2024, random_strength=33, bagging_temperature=1, max_ctr_complexity=2
# 73.72% | iterarions=500, learning_rate=0.208787, depth=10, l2_leaf_reg=1.25048, random_strength=15, bagging_temperature=1, max_ctr_complexity=2

# Procederemos a entrenar modelos CatBoost en base a estas tres combinaciones de hiperparámetros obtenidas
# para determinar cual de ellas presenta mejores resultados al clasificar nuestros datos.

# Identificación de variables categoricas
categorical_features_indices = np.where(data.dtypes == np.object)[0]
train_pool = Pool(X_train, y_train, cat_features = categorical_features_indices)
test_pool = Pool(X_test, y_test, cat_features = categorical_features_indices)

# Para la primera combinación
cb_1a = CatBoostClassifier(iterations=600, learning_rate=0.166129, depth=7, l2_leaf_reg=0.963535, random_strength=11,
                            bagging_temperature=1, max_ctr_complexity=0, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_1a.fit(train_pool, eval_set = test_pool)
y_pred_1a = cb_1a.predict(X_test)

# Para la segunda combinación
cb_1b = CatBoostClassifier(iterations=1200, learning_rate=0.246729, depth=9, l2_leaf_reg=7.2024, random_strength=33,
                            bagging_temperature=1, max_ctr_complexity=2, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_1b.fit(train_pool, eval_set = test_pool)
y_pred_1b = cb_1b.predict(X_test)

# Para la tercera combinación
cb_1c = CatBoostClassifier(iterations=500, learning_rate=0.208787, depth=10, l2_leaf_reg=1.25048, random_strength=15,
                            bagging_temperature=1, max_ctr_complexity=2, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_1c.fit(train_pool, eval_set = test_pool)
y_pred_1c = cb_1c.predict(X_test)


# COMPARACIÓN DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinación
f1_1a = f1_score(y_test, y_pred_1a)
acc_1a = accuracy_score(y_test, y_pred_1a)
auc_1a = roc_auc_score(y_test, y_pred_1a)
report_1a = classification_report(y_test,y_pred_1a)

# Para la segunda combinación
f1_1b = f1_score(y_test, y_pred_1b)
acc_1b = accuracy_score(y_test, y_pred_1b)
auc_1b = roc_auc_score(y_test, y_pred_1b)
report_1b = classification_report(y_test,y_pred_1b)

# Para la tercera combinación
f1_1c = f1_score(y_test, y_pred_1c)
acc_1c = accuracy_score(y_test, y_pred_1c)
auc_1c = roc_auc_score(y_test, y_pred_1c)
report_1c = classification_report(y_test,y_pred_1c)

# A continuación visualizaremos el puntaje de la métrica F1 y la precisión para cada combinación, a la vez que
# tambien observaremos un reporte de las principales métricas para evaluar la capacidad de clasificación de
# nuestros modelos

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

# En principio, observamos que la segunda combinacion es la que presenta valores de métrica mayores que las
# demas combinaciones, aunque la diferencia entre ellos es muy minima.

# Procederemos a graficar la matriz de confusión y la curva ROC-AUC.

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_1a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACIÓN 1",fontsize=14)

sns.heatmap(confusion_matrix(y_test, y_pred_1b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACIÓN 2",fontsize=14)

sns.heatmap(confusion_matrix(y_test, y_pred_1c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACIÓN 3",fontsize=14)

plt.show()

# Con respecto a las matrices de confusion, observamos que la combinacion 2 presenta un ratio mejor equilibrado
# al momento de predecir correctamente si un cliente solicita un deposito a plazo fijo o no, a la vez que 
# tambien observamos que la combinacion 3 es la que peores resultado presenta al realizar esta clasificacion. Es
# por ello que tomaremos en cuenta a la combinacion 2 como la que mejores resultados arrojo en este apartado de
# evaluacion.

y_pred_prob1a = cb_1a.predict_proba(X_test)[:,1]
fpr_1a, tpr_1a, thresholds_1a = roc_curve(y_test, y_pred_prob1a)
y_pred_prob1b = cb_1b.predict_proba(X_test)[:,1]
fpr_1b, tpr_1b, thresholds_1b = roc_curve(y_test, y_pred_prob1b)
y_pred_prob1c = cb_1c.predict_proba(X_test)[:,1]
fpr_1c, tpr_1c, thresholds_1c = roc_curve(y_test, y_pred_prob1c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_1a, tpr_1a, label='Combinación 1',color = "r")
plt.plot(fpr_1b, tpr_1b, label='Combinación 2',color = "g")
plt.plot(fpr_1c, tpr_1c, label='Combinación 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Del grafico de la curva ROC-AUC no podemos diferenciar claramente si la combinacion 1 o 2 es la que mejor tasa
# de verdaderos positivos (VP) y falsos positivos (FP) tiene, sin embargo, podemos observar que la curva de la
# combinacion 3 tiende a ser menor comparado con las demas combinaciones, por lo que combinado con los resultados
# de las metricas anteriores, podemos ir descartando esta combinacion.

print("AUC primera comb.: %.2f%%" % (auc_1a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_1b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_1c * 100.0))

# Por ultimo, podemos ver que el valor de la metrica AUC nos da claridad al momento de decidir que combinacion
# presenta una mejor tasa de VP y FP, ya que como habiamos deducido anteriormente, la combinacion 3 es la que
# peores resultados arroja, y que tanto la combinacion 1 como la combinacion 2 presentan resultados similares,
# sin embargo, la combinacion 2 presenta una ligera superioridad comparado con las demas combinaciones. Entonces,
# uniendo los resultados de las metricas anteriormente vistas, podemos concluir que el modelo construido con la
# combinacion 2 es el que mejor clasifica estos datos, por lo tanto, utilizaremos este modelo como referente
# del conjunto de "Hiperparametros para datos con outliers y sin codificacion manual".
 

#------------------------------
# HIPERPARÁMETROS PARA DATOS CON OUTLIERS Y CON CODIFICACION MANUAL

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
    
    train_pool = Pool(X_train_cod, y_train_cod)
    test_pool = Pool(X_test_cod, y_test_cod)
    
    # Inicialización y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluación y obtención de métricas
    preds = model.predict(X_test_cod)
    metric = accuracy_score(y_test_cod, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_2 = study.trials_dataframe()

# Se ejecutó la función tres veces de forma independiente, y posterior a ello, se registro
# la mejor combinación de parámetros que arrojo cada ejecución, siendo estas las siguientes:

# 74.29% | iterarions=900, learning_rate=0.198706, depth=9, l2_leaf_reg=4.72514, random_strength=40, bagging_temperature=0, max_ctr_complexity=3
# 74.23% | iterarions=1000, learning_rate=0.0686307, depth=7, l2_leaf_reg=6.87847, random_strength=2, bagging_temperature=1, max_ctr_complexity=1
# 74.17% | iterarions=500, learning_rate=0.103597, depth=11, l2_leaf_reg=7.95198, random_strength=8, bagging_temperature=0, max_ctr_complexity=4

# Procederemos a entrenar modelos CatBoost en base a estas tres combinaciones de hiperparámetros obtenidas
# para determinar cual de ellas presenta mejores resultados al clasificar nuestros datos.

train_pool = Pool(X_train_cod, y_train_cod)
test_pool = Pool(X_test_cod, y_test_cod)

# Para la primera combinación
cb_2a = CatBoostClassifier(iterations=900, learning_rate=0.198706, depth=9, l2_leaf_reg=4.72514, random_strength=40,
                            bagging_temperature=0, max_ctr_complexity=3, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_2a.fit(train_pool, eval_set = test_pool)
y_pred_2a = cb_2a.predict(X_test_cod)

# Para la segunda combinación #Utilizando random seed
cb_2b = CatBoostClassifier(iterations=1000, learning_rate=0.0686307, depth=7, l2_leaf_reg=6.87847, random_strength=2,
                            bagging_temperature=1,  max_ctr_complexity=1, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_2b.fit(train_pool, eval_set = test_pool)
y_pred_2b = cb_2b.predict(X_test_cod)

# Para la tercera combinación
cb_2c = CatBoostClassifier(iterations=500, learning_rate=0.103597, depth=11, l2_leaf_reg=7.95198, random_strength=8,
                            bagging_temperature=0, max_ctr_complexity=4, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_2c.fit(train_pool, eval_set = test_pool)
y_pred_2c = cb_2c.predict(X_test_cod)


# COMPARACIÓN DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinación
f1_2a = f1_score(y_test_cod, y_pred_2a)
acc_2a = accuracy_score(y_test_cod, y_pred_2a)
auc_2a = roc_auc_score(y_test_cod, y_pred_2a)
report_2a = classification_report(y_test_cod,y_pred_2a)

# Para la segunda combinación
f1_2b = f1_score(y_test_cod, y_pred_2b)
acc_2b = accuracy_score(y_test_cod, y_pred_2b)
auc_2b = roc_auc_score(y_test_cod, y_pred_2b)
report_2b = classification_report(y_test_cod,y_pred_2b)

# Para la tercera combinación
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

# Se puede observar que si bien todos los valores de métrica para todas las combinaciones que tenemos
# son muy similares, la tercera combinacion es la que destaca por un pequeño margen porcentual de las
# demas. Mientras que la primera combinacion parece tener ligeramente un rendimiento menor en comparacion
# con las demas combinaciones

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACIÓN 1",fontsize=14)

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACIÓN 2",fontsize=14)

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACIÓN 3",fontsize=14)

plt.show()

# De las matrices de confusion podemos reafirmar nuestra suposicion al observar que la combinacion 3
# es la que presenta un mejor ratio de VP y FP, mientras que la combinacion 1 esta por debajo de las
# demas combinaciones, lo cual nos da motivos para poder ir descartando el modelo construido con esta
# ultima combinacion.

y_pred_prob2a = cb_2a.predict_proba(X_test_cod)[:,1]
fpr_2a, tpr_2a, thresholds_2a = roc_curve(y_test_cod, y_pred_prob2a)
y_pred_prob2b = cb_2b.predict_proba(X_test_cod)[:,1]
fpr_2b, tpr_2b, thresholds_2b = roc_curve(y_test_cod, y_pred_prob2b)
y_pred_prob2c = cb_2c.predict_proba(X_test_cod)[:,1]
fpr_2c, tpr_2c, thresholds_2c = roc_curve(y_test_cod, y_pred_prob2c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_2a, tpr_2a, label='Combinación 1',color = "r")
plt.plot(fpr_2b, tpr_2b, label='Combinación 2',color = "g")
plt.plot(fpr_2c, tpr_2c, label='Combinación 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Con respecto a la curva ROC-AUC no se puede visualizar una clara diferencia entre las curvas de la
# combinacion 2 y 3, sin embargo, se puede observar que la curva de la combinacion 1 parece estar
# por debajo en comparacion con el de las demas combinaciones, lo cual nos dice que no tiene un buen
# ratio comparandola con las demas al momento de clasificar una muestra con la clase que le corresponde
# (VP, FP).

print("AUC primera comb.: %.2f%%" % (auc_2a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_2b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_2c * 100.0))

# Por ultimo, con un valor porcentual de la métrica de la curva, podemos tomar una decision final
# respecto a que combinacion elegir. Tomar esta decision no sera muy dificil, ya que en 3 de las 4
# pruebas de evaluacion observamos claramente que la combinacion 3 es la que presento mejores resultados
# en comparacion con las demas combinaciones, aunque la diferencia entre estas sea porcentual, es por 
# ello que el modelo construido con esta combinacion sera usado como referente del conjunto de
# "Hiperparametros para datos con outliers y con codificacion manual".


#------------------------------
# HIPERPARÁMETROS PARA DATOS SIN OUTLIERS Y SIN CODIFICACION MANUAL

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
    
    # Identificación de variables categoricas
    categorical_features_indices = np.where(data2.dtypes == np.object)[0]
    
    train_pool = Pool(X2_train, y2_train, cat_features = categorical_features_indices)
    test_pool = Pool(X2_test, y2_test, cat_features = categorical_features_indices)
    
    # Inicialización y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluación y obtención de métricas
    preds = model.predict(X2_test)
    metric = accuracy_score(y2_test, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_3 = study.trials_dataframe()

# Se ejecutó la función tres veces de forma independiente, y posterior a ello, se registró la mejor combinación
# de parámetros que arrojó cada ejecución, siendo estas las siguientes:

# 75.84% | iterarions=900, learning_rate=0.118849, depth=7, l2_leaf_reg=9.48661, random_strength=15, bagging_temperature=1, max_ctr_complexity=6
# 75.72% | iterarions=1000, learning_rate=0.115247, depth=10, l2_leaf_reg=7.87387, random_strength=19, bagging_temperature=1, max_ctr_complexity=6
# 75.66% | iterarions=700, learning_rate=0.0431437, depth=10, l2_leaf_reg=7.91287, random_strength=14, bagging_temperature=0, max_ctr_complexity=4

# Procederemos a entrenar un nuevo modelo XGBoost en base a las tres combinaciones de hiperparámetros
# obtenidas para determinar cual de ellas presenta mejores resultados al clasificar nuestros datos

# Identificación de variables categoricas
categorical_features_indices = np.where(data2.dtypes == np.object)[0]
train_pool = Pool(X2_train, y2_train, cat_features = categorical_features_indices)
test_pool = Pool(X2_test, y2_test, cat_features = categorical_features_indices)

# Para la primera combinación
cb_3a = CatBoostClassifier(iterations=900, learning_rate=0.118849, depth=7, l2_leaf_reg=9.48661, random_strength=15,
                            bagging_temperature=1, max_ctr_complexity= 6, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_3a.fit(train_pool, eval_set = test_pool)
y_pred_3a = cb_3a.predict(X2_test)

# Para la segunda combinación #Utilizando random seed
cb_3b = CatBoostClassifier(iterations=1000, learning_rate=0.115247, depth=10, l2_leaf_reg=7.87387, random_strength=19,
                            bagging_temperature=1,  max_ctr_complexity= 6, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_3b.fit(train_pool, eval_set = test_pool)
y_pred_3b = cb_3b.predict(X2_test)

# Para la tercera combinación
cb_3c = CatBoostClassifier(iterations=700, learning_rate=0.0431437, depth=10, l2_leaf_reg=7.91287, random_strength=14,
                            bagging_temperature=0, max_ctr_complexity= 4, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_3c.fit(train_pool, eval_set = test_pool)
y_pred_3c = cb_3c.predict(X2_test)


# COMPARACIÓN DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinación
f1_3a = f1_score(y2_test, y_pred_3a)
acc_3a = accuracy_score(y2_test, y_pred_3a)
auc_3a = roc_auc_score(y2_test, y_pred_3a)
report_3a = classification_report(y2_test,y_pred_3a)

# Para la segunda combinación
f1_3b = f1_score(y2_test, y_pred_3b)
acc_3b = accuracy_score(y2_test, y_pred_3b)
auc_3b = roc_auc_score(y2_test, y_pred_3b)
report_3b = classification_report(y2_test,y_pred_3b)

# Para la tercera combinación
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

# Observamos demasiada simulitud entre los puntajes obtenidos por las 3 combinaciones, donde la combinacion que
# sobresale en comparacion de las demas en puntaje F1, no lo hace en Accuracy, es por ello que necesitamos mas
# indicios (metricas de evaluacion) que nos indiquen que combinacion se adapta mejor a nuestros datos.

# Procederemos a graficar la matriz de confusión y la curva ROC-AUC.

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y2_test, y_pred_3a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACIÓN 1",fontsize=14)

sns.heatmap(confusion_matrix(y2_test, y_pred_3b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACIÓN 2",fontsize=14)

sns.heatmap(confusion_matrix(y2_test, y_pred_3c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACIÓN 3",fontsize=14)

plt.show()

# De las matrices de confusion observamos que la combinacion 2 es el que mejor ratio en verdaderos positivos 
# tiene, sin embargo, es muy inferior en comparacion con las demas combinaciones con respecto a los falsos
# positivos. Una comportamiento similar se observa en la combinacion 3, donde la situacion es la misma pero
# invertida, donde el mejor ratio de prediccion se lo llevan los falsos positivos y el peor los verdaderos
# positivos. Con respecto a la combinacion 1 podemos observar un mejor balance entre ambos ratios de prediccion,
# puesto que la cantidad de muestras catalogadas correctamente parece estar en un termino medio entre las demas
# combinaciones, es por ello que en el area de las matrices de confusion preferimos la combinacion 1.

y_pred_prob3a = cb_3a.predict_proba(X2_test)[:,1]
fpr_3a, tpr_3a, thresholds_3a = roc_curve(y2_test, y_pred_prob3a)
y_pred_prob3b = cb_3b.predict_proba(X2_test)[:,1]
fpr_3b, tpr_3b, thresholds_3b = roc_curve(y2_test, y_pred_prob3b)
y_pred_prob3c = cb_3c.predict_proba(X2_test)[:,1]
fpr_3c, tpr_3c, thresholds_3c = roc_curve(y2_test, y_pred_prob3c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_3a, tpr_3a, label='Combinación 1',color = "r")
plt.plot(fpr_3b, tpr_3b, label='Combinación 2',color = "g")
plt.plot(fpr_3c, tpr_3c, label='Combinación 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Con respecto a la curva ROC-AUC podemos identificar que aparentemente la combinacion 2 y 3 son las que presentan
# un mejor ratio en VP y FP, sin embargo, la diferencia no es tan clara con respecto a la combinacion 1 puesto
# que hay momentos en donde presentan una inclinicacion similar, es por ello que calcularemos el valor de su
# metrica con el fin de poder tener una mejor interpretabilidad.

print("AUC primera comb.: %.2f%%" % (auc_3a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_3b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_3c * 100.0))

# Con esto ultimo, observamos que el mejor valor AUC se lo lleva la combinacion 3, el peor valor la combinacion
# 2, y en termino medio se encuentra la combinacion 1. No obstante, la diferencia entre estos valores es muy
# pequeña, por lo cual no podemos decir que una combinacion mucho mas efectiva que otra. Entonces, uniendo los
# resultados de todas las metricas vistas, podemos considerar a todas las combinaciones como buenas, ya que
# no existe una gran diferencia en efectividad de prediccion entre ellas, sin embargo, en esta ocacion
# eligiremos la combinacion 1 como ganadora, ya que es la que presenta un equilibrio entre predecir correctamente
# aquellas muestras que son positivas (deposito a plazo fijo) y negativas (no deposito a plazo fijo), por lo 
# tanto, utilizaremos el modelo construido con esta combinacion como referente del conjunto de "Hiperparametros
# para datos sin outliers y sin codificacion manual".


#------------------------------
# HIPERPARÁMETROS PARA DATOS SIN OUTLIERS Y CON CODIFICACION MANUAL

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
    
    train_pool = Pool(X2_train_cod, y2_train_cod)
    test_pool = Pool(X2_test_cod, y2_test_cod)
    
    # Inicialización y entrenamiento del modelo
    model = CatBoostClassifier(**params) 
    model.fit(train_pool, eval_set=test_pool, verbose=True)
    
    # Evaluación y obtención de métricas
    preds = model.predict(X2_test_cod)
    metric = accuracy_score(y2_test_cod, preds)
    
    return metric


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=70)

print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))
best_4 = study.trials_dataframe()

# Se ejecutó la función tres veces de forma independiente, y posterior a ello, se registró la mejor combinación
# de parámetros que arrojó cada ejecución, siendo estas las siguientes:

# 75.24% | iterarions=600, learning_rate=0.0856975, depth=9, l2_leaf_reg=7.42101, random_strength=0, bagging_temperature=0, max_ctr_complexity=10
# 75.18% | iterarions=1200, learning_rate=0.0274008, depth=10, l2_leaf_reg=7.42817, random_strength=0, bagging_temperature=1, max_ctr_complexity=1
# 75.15% | iterarions=700, learning_rate=0.0854912, depth=10, l2_leaf_reg=5.43813, random_strength=0, bagging_temperature=0, max_ctr_complexity=10

# Procederemos a entrenar un nuevo modelo XGBoost en base a las tres combinaciones de hiperparámetros
# obtenidas para determinar cual de ellas presenta mejores resultados al clasificar nuestros datos

train_pool = Pool(X2_train_cod, y2_train_cod)
test_pool = Pool(X2_test_cod, y2_test_cod)

# Para la primera combinación
cb_4a = CatBoostClassifier(iterations=600, learning_rate=0.0856975, depth=9, l2_leaf_reg=7.42101, random_strength=0,
                            bagging_temperature=0, max_ctr_complexity=10, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_4a.fit(train_pool, eval_set = test_pool)
y_pred_4a = cb_4a.predict(X2_test_cod)

# Para la segunda combinación #Utilizando random seed
cb_4b = CatBoostClassifier(iterations=1200, learning_rate=0.0274008, depth=10, l2_leaf_reg=7.42817, random_strength=0,
                            bagging_temperature=1,  max_ctr_complexity=1, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_4b.fit(train_pool, eval_set = test_pool)
y_pred_4b = cb_4b.predict(X2_test_cod)

# Para la tercera combinación
cb_4c = CatBoostClassifier(iterations=700, learning_rate=0.0854912, depth=10, l2_leaf_reg=5.43813, random_strength=0,
                            bagging_temperature=0, max_ctr_complexity=10, auto_class_weights= "Balanced", loss_function = "Logloss",
                            eval_metric = "AUC", task_type= "GPU", use_best_model= True, random_seed=42)

cb_4c.fit(train_pool, eval_set = test_pool)
y_pred_4c = cb_4c.predict(X2_test_cod)


# COMPARACIÓN DE RENDIMIENTO ENTRE COMBINACIONES

# Para la primera combinación
f1_4a = f1_score(y2_test_cod, y_pred_4a)
acc_4a = accuracy_score(y2_test_cod, y_pred_4a)
auc_4a = roc_auc_score(y2_test_cod, y_pred_4a)
report_4a = classification_report(y2_test_cod,y_pred_4a)

# Para la segunda combinación
f1_4b = f1_score(y2_test_cod, y_pred_4b)
acc_4b = accuracy_score(y2_test_cod, y_pred_4b)
auc_4b = roc_auc_score(y2_test_cod, y_pred_4b)
report_4b = classification_report(y2_test_cod,y_pred_4b)

# Para la tercera combinación
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

# Respecto a esta combinacion de metricas se observa claramente que la tercera combinacion es la que mejores
# resultados presenta, aunque la diferencia de sus puntajes comparados con el de las demas combinaciones sea
# relativamente pequeña. Tambien podemos observar que los puntajes de la primera combinacion son los que menores
# valores presentaron.

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4a), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0])
ax[0].set_title("COMBINACIÓN 1",fontsize=14)

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1])
ax[1].set_title("COMBINACIÓN 2",fontsize=14)

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[2])
ax[2].set_title("COMBINACIÓN 3",fontsize=14)

plt.show()

# Las matrices de confusion dejan en clara evidencia la superioridad predictora del modelo cosntruido con la
# tercera combinacion, ya que este presenta un mejor ratio en la clasificacion de muestras como VP y FP en 
# comparacion con las demas. Estas matrices tambien nos permiten ir concluyendo que la primera combinacion tiene
# un rendimiento predictivo inferior en comparacion con las demas, ya que presenta peores resultados de clasificacion

y_pred_prob4a = cb_4a.predict_proba(X2_test_cod)[:,1]
fpr_4a, tpr_4a, thresholds_4a = roc_curve(y2_test_cod, y_pred_prob4a)
y_pred_prob4b = cb_4b.predict_proba(X2_test_cod)[:,1]
fpr_4b, tpr_4b, thresholds_4b = roc_curve(y2_test_cod, y_pred_prob4b)
y_pred_prob4c = cb_4c.predict_proba(X2_test_cod)[:,1]
fpr_4c, tpr_4c, thresholds_4c = roc_curve(y2_test_cod, y_pred_prob4c)

plt.figure(figsize=(16, 8))
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_4a, tpr_4a, label='Combinación 1',color = "r")
plt.plot(fpr_4b, tpr_4b, label='Combinación 2',color = "g")
plt.plot(fpr_4c, tpr_4c, label='Combinación 3',color = "b")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# Con respecto a la curva ROC-AUC, no podemos identificar claramente que combinacion es superior a las demas, ya
# que existen algunos trazos en los que una combinacion es inferior a otra, y otros en los que es superior a las
# demas, es por ello que calcularemos el valor de su metrica con el fin de poder tener una mejor interpretabilidad

print("AUC primera comb.: %.2f%%" % (auc_4a * 100.0))
print("AUC segunda comb.: %.2f%%" % (auc_4b * 100.0))
print("AUC tercera comb.: %.2f%%" % (auc_4c * 100.0))

# Estos resultados indican que la tercera combinacion posee un mejor ratio en la correcta prediccion de nuestros
# datos, y que como ya sospechabamos anteriormente, la primera combinacion tiene el peor rendimiento predictivo en 
# comparacion con las demas. Es por ello que uniendo estos resultados con el de las demas metricas anteriormente
# vistas, facilmente podemos concluir que el modelo construido con la tercera combinacion es el que mejor clasifica
# nuestros datos, por lo tanto, sera usado como referente del conjunto de "Hiperparametros para datos sin outliers
# y con codificacion manual".


#-----------------------------
# ELECCIÓN DEL MEJOR MODELO
#-----------------------------

# Después de haber elegido las cuatro mejores combinaciones en base al entrenamiento de conjuntos con diferentes
# tipos de transformación y codificacion, procederemos a compararlos entre sí para quedarnos con un modelo definitivo
# el cual mejores resultados de evaluación tenga.

print("F1 Primer conjunto: %.2f%%" % (f1_1b * 100.0))
print("Accuracy Primer conjunto: %.2f%%" % (acc_1b * 100.0))
print("-------------------------------")
print("F1 Segundo conjunto: %.2f%%" % (f1_2c * 100.0))
print("Accuracy Segundo conjunto: %.2f%%" % (acc_2c * 100.0))
print("-------------------------------")
print("F1 Tercer conjunto: %.2f%%" % (f1_3b * 100.0))
print("Accuracy Tercer conjunto: %.2f%%" % (acc_3b * 100.0))
print("-------------------------------")
print("F1 Cuarto conjunto: %.2f%%" % (f1_4c * 100.0))
print("Accuracy Cuarto conjunto: %.2f%%" % (acc_4c * 100.0))

print(report_1b)
print("-------------------------------------------------")
print(report_2c)
print("-------------------------------------------------")
print(report_3b)
print("-------------------------------------------------")
print(report_4c)

# De principio estamos observando que el modelo del primer conjunto (Datos rebalanceados por XGBoost) tiene
# un rendimiento superior en cuanto a puntaje F1 se refiere, y en cuanto a precisión tiene un puntaje similar
# con el modelo de la combinación 2 (Datos rebalanceados por SMOTE-NC).

fig, ax = plt.subplots(2, 2, figsize=(20, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_1b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0,0])
ax[0][0].set_title("PRIMER CONJUNTO",fontsize=14)

sns.heatmap(confusion_matrix(y_test_cod, y_pred_2c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[0,1])
ax[0][1].set_title("SEGUNDO CONJUNTO",fontsize=14)

sns.heatmap(confusion_matrix(y2_test, y_pred_3b), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1,0])
ax[1][0].set_title("TERCER CONJUNTO",fontsize=14)

sns.heatmap(confusion_matrix(y2_test_cod, y_pred_4c), annot=True, fmt = "d", linecolor="k", linewidths=3, ax=ax[1,1])
ax[1][1].set_title("CUARTO CONJUNTO",fontsize=14)

plt.show()

# De nuestras matrices de confusión observamos que el modelo del primer conjunto tiene ligeramente una mayor
# sensibilidad en comparación con el del segundo conjunto, y que el modelo del tercer conjunto tiene un bajo
# rendimiento en la identificación de verdaderos positivos.

plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_1b, tpr_1b, label='Primer conjunto',color = "r")
plt.plot(fpr_2c, tpr_2c, label='Segundo conjunto',color = "g")
plt.plot(fpr_3b, tpr_3b, label='Tercer conjunto',color = "b")
plt.plot(fpr_4c, tpr_4c, label='Cuarto conjunto',color = "y")
plt.xlabel('Ratio de Falsos Positivos')
plt.ylabel('Ratio de Verdaderos Positivos')
plt.title('Curva ROC-AUC',fontsize=16)
plt.legend()
plt.show()

# El gráfico de la curva ROC-AUC nos da un resultado muy interesante, ya que podemos ver la superioridad del
# modelo del primer conjunto en cuanto a la predicción correcta de verdaderos positivos y falsos positivos en
# comparación con los demás modelos, por lo tanto, ya se puede deducir cual es la combinación de parámetros
# que mejor se ajustan a nuestros datos.

print("AUC Primer conjunto: %.2f%%" % (auc_1b * 100.0))
print("AUC Segundo conjunto: %.2f%%" % (auc_2c * 100.0))
print("AUC Tercer conjunto: %.2f%%" % (auc_3b * 100.0))
print("AUC Cuarto conjunto: %.2f%%" % (auc_4c * 100.0))

# Finalmente, con estos puntajes calculados, llegamos a la decision de utilizar el mejor modelo proveniente
# del primer conjunto (Datos rebalanceados por XGBoost), debido a que a lo largo de todo el proceso de
# selección, mostró superioridad frente a los demás modelos.

# Combinación de parámetros del modelo final:
# tree_method="gpu_hist", objective="binary:logistic", eval_metric="auc", use_label_encoder=False,
# n_estimators=400, max_depth=18, learning_rate=0.0013, subsample=0.2, colsample_bytree=0.9, seed=21

# Guardado del modelo
joblib.dump(xgb_1c, "XGboost_Model_Churn")




