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


# El conjunto de datos con el que vamos a tratar almacena características de 11162 personas a los que un banco
# contacto para ofrecerles el servicio de deposito a plazo fijo, e indica si estos al final decidieron adquirir
# dicho servicio o no.
data = pd.read_csv("Bank_Marketing.csv")


#------------------------------------------------------------------------------------------------------------
#                                          EXPLORACIÓN DE LOS DATOS
#------------------------------------------------------------------------------------------------------------

data.head()

data.shape

data.info()

#------------------------------------------
#  ELIMINACIÓN Y CODIFICACIÓN DE VARIABLES
#------------------------------------------

# Hay que tener en cuenta algo de suma importancia en nuestros datos, y es que la variable "contact" hace
# referencia al tiempo de duracion en segundos del ultimo contacto que se realizo con la persona antes que
# decidiera adquirir o no un deposito a plazo fijo, y como naturalmente este valor no se conoce hasta despues
# de haber realizado la llamada que es cuando ya se sabe la decision de la persona, se procedera a eliminar
# al momento de construir nuestro modelo predictivo, puesto que estaria otorgando informacion que de por si
# no se conoce de antemano










