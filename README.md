# Proyecto Gestión de Datos
## Análisis y Visualización de Tendencias Epidemiológicas Globales con Python y Pandas
### Introducción
Este proyecto consistió en examinar y representar de manera visual la progresión de las tendencias epidemiológicas a nivel mundial de la pandemia del COVID-19, utilizando Python y sus principales herramientas para ciencias de datos. Para lo anterior, se utilizó el [repositorio de datos públicos del COVID-19](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data), que fue recolectado y distribuido por el Center for Systems Science and Engineering (CSSE) de la Universidad Johns Hopkins (JHU).

El resultado final de este trabajo se puede visualizar en un dashboard interactivo realizado con Streamlit, el cual se puede visualizar a continuación:

* **[Acceder a Dashboard Interactivo](https://proyectogestiondedatos-xhqdskk23almdwrtdmnwaw.streamlit.app/)**

### Instrucciones
Una vez dentro del dashboard, se pueden visualizar en la barra lateral izquierda, los filtros que se pueden utilizar dentro del mismo, los cuales son:

* Filtro por continente (África, Asia, América, Europa, entre otros). Se puede seleccionar ninguno, uno o más de ser necesario.
* Filtro por país (Argentina, Brasil, Chile, Colombia, entre otros). Se puede seleccionar ninguno, uno o más de ser necesario
* Filtro por rango de fechas. El rango global de datos disponibles abarca del 2020/01/22 al 2023/03/10. Las fechas se seleccionan por medio de un calendario, por lo que el usuario no debe escribir manualmente las fechas.

Por otro lado, en la vista central del dashboard, podremos visualizar diversos indicadores y gráficos que nos entregarán información útil del conjunto de datos filtrado de acuerdo a los criterios anteriores. Estos indicadores y gráficos son (en orden de arriba hacia abajo):

* **Indicadores Clave:** Esto nos muestra inmediatamente la cantidad de casos confirmados, activos, recuperados, y fallecidos, hasta la fecha indicada y acotada a los límites geográficos en caso de haber definido esos filtros.
* **Evolución Temporal:** Esto nos muestra un gráfico de líneas que permite observar la evolución temporal de los casos confirmados, activos, recuperados, y fallecidos a lo largo del rango temporal de datos definido. Podemos hacer zoom o seleccionar una zona del gráfico en caso de querer visualizar algo con mayor detalle.
* **Nuevos Casos Diarios:** Esto nos muestra un gráfico de barras que permite consultar los nuevos casos que se registraron de forma diaria (distinto a los casos confirmados, que se van acumulando a lo largo del tiempo). Al igual que en el caso anterior, podemos hacer zoom o seleccionar una zona del gráfico en caso de querer visualizar algo con mayor detalle.
* **Distribución por País:** Esto nos muestra un gráfico de barras apiladas, el cual nos permite visualizar la proporción de casos confirmados, activos, recuperados y fallecidos por país. Al igual que en los casos anteriores, podemos hacer zoom o seleccionar una zona del gráfico en caso de querer visualizar algo con mayor detalle.
* **Análisis de Crecimiento:** Esto nos muestra dos indicadores, el primero es el *Ratio de Rebrote* (razón entre los casos confirmados en el día 14 versus los 14 días previos) y el segundo es la *Tasa de Crecimiento Diaria* (promedio de los 7 días anteriores al último del rango de fechas). Ambas variables permiten proyectar si existió o no un rebrote significativo en el rango temporal y geográfico definido, lo que se traduce en el mensaje de alerta o informativo ubicado debajo de estos indicadores.
* **Análisis Automático:** Esto nos muestra otros datos relevantes calculados a partir de los datos filtrados temporal y geográficamente, el primero de ellos son los *Nuevos Casos en el Rango Seleccionado* que contabiliza cuantos nuevos casos se registraron en el periodo, el segundo es el porcentaje de aumento o disminución de nuevos casos respecto a los últimos 14 días, y por último el país con más nuevos casos acotado al rango geográfico definido.