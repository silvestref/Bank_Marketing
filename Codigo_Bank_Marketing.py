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
from matplotlib.ticker import FormatStrFormatter
import association_metrics as am

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

# Debido a que las predicciones hechas por los bosques aleatorios se basan en el promedio del resultado de 
# varios arboles de decision, tendremos algunos datos imputados como decimal en variables que son enteras, como
# en el caso de "age", es por ello que redondearemos dichos valores decimales en cada variable que solo 
# contenga valores enteros

for col in ["age", "day", "campaign", "previous", "balance", "duration"]:
    data2[col] = data2[col].round()

# Ahora si podemos graficar para observar el cambio en nuestros datos despues de la imputacion

fig, ax = plt.subplots(1, 3, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data2[["age", "day", "campaign", "previous"]], palette="Set3")
sns.boxplot(ax=ax[1], data= data2[["balance"]], palette="Pastel1")
sns.boxplot(ax=ax[2], data= data2[["duration"]], palette="Pastel1")
plt.show()

# Del grafico podemos observar que que todas las variables a excepcion de "balance" y "duration" estan libres
# de outliers.

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data2[["balance"]], palette="Set3")
sns.distplot(data2["balance"], ax=ax[1])

fig, ax = plt.subplots(1, 2, figsize=(14,7))
sns.boxplot(ax=ax[0], data= data2[["duration"]], palette="Set3")
sns.distplot(data2["duration"], ax=ax[1])

# Analizando las variables que aun tienen presencia de valores atipicos, se ve que la varianza en la distribucion
# de estos valores ya no es tan extrema como teniamos inicialmente, si no que ahora se distribuyen en un rango
# menor a 1000 unidades, incluso pudiendose acercar a una distribucion normal.

Q1 = data2.quantile(0.25)
Q3 = data2.quantile(0.75)
IQR = Q3 - Q1

( (data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR)) ).sum() / data2.shape[0] * 100

# A la vez que tambien observamos que estos datos atipicos solo constituyen el 5.6% y 4.1% respectivamente 
# del total, lo cual es una cifra moderadamente baja. Entonces podemos tomar dos decisiones, elimnarlos o
# conservarlos como parte de nuestros datos. En esta ocacion, eligire conservarlos ya que pueden contener
# informacion util para el analisis y para el modelo de clasificacion, ademas que su presencia es relativamente
# baja con respecto del total y su distancia de los extremos no es tan alarmante ni exagerada.


#------------------------------------
# IDENTIFICACIÓN DE VALORES FALTANTES 
#------------------------------------

# Observamos cuantos valores faltantes hay en nuestro conjunto de datos
data2.isnull().sum().sum()

# Debido a que no hay presencia de valores faltantes o nulos, no sera necesario tomar acciones al respecto



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


# CORRELACION DE PEARSON

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


# V DE CRAMER

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




