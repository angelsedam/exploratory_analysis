
import numpy as np
import pandas as pd
import os
import plotly as py
import plotly.figure_factory as FF
from plotly import tools
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.features import DivIcon
from folium.plugins import HeatMap
import warnings
import datetime
import squarify
warnings.filterwarnings("ignore")



os.chdir('/Users/angelseda/Documents/angel/exploratory_analysis/data')
df = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
#print(df.dtypes)
print(df.head())
#Muesrta las dimenciones del df numero de registros y columnas
print(df.shape)

#Revisar el numero de valores nulos en el df
print(df.isnull().sum())

#Crear un df donde no haya valoes nulso en la columna precio
no_null_df =df[df['Price'].notnull()]
print(no_null_df.isnull().sum())

#Creamos una copia del df que no tienen valores nulos
db=no_null_df.copy()
print(db.head())

#Crear una nueva columna
#Extraer el año de una fecha completa
db['year'] = pd.DatetimeIndex(db['Date']).year
#print(db['year'].head())

#Contrar cuantas casas hay por region
region = db['Regionname'].value_counts().sort_values(ascending=True)
print(region)

#establecemos el formato de los gráficos....en matplotlib puede encontrar los diferentes
#estilos disponibles: default, classic, Solarize_light2, classic_teste_patch,bmh,dark_background, fast,ggplot, grayscale, etc.
#https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

#Usando matplot
# plt.style.use('fivethirtyeight')
fig,ax =plt.subplots()
region.plot.barh(stacked=True, figsize=(4,2), title='Numero de casas por region', color='k', alpha=0.8)
ax.set_ylabel('Region')
ax.set_xlabel('Numero de casas')
#plt.show()

#Usando pandas
plot=pd.value_counts(db['Regionname']).plot(kind='barh',title='Numero de casas por region', rot=0)
plot.set_ylabel('Region')
plot.set_xlabel('Numero de casas')
#plt.show()

fig,ax =plt.subplots()
types = db['Type'].value_counts()
types.plot.bar(figsize =(4,2), rot=0, align='center', color = 'orange')
ax.set_title('Numero de casas por tipo', size=12)
ax.set_ylabel('Numero de casas',size=10)
ax.set_xlabel('Tipos de casa',size=10)
#plt.show()

plt.figure(figsize=(4,4))
fig,ax =plt.subplots()
sns.regplot(x='Rooms',y='Price',data=db)
ax.set_title('Rooms vs Price', size=12)
ax.set_ylabel('Precio', size=10)
ax.set_xlabel('Rooms',size=10)
#plt.show()

plt.figure(figsize=(4,4))
fig,ax=plt.subplots()
sns.regplot(x='Distance', y='Price', data=db)
ax.set_title('Precio vs Distancia al CBD (kms)', size=12)
ax.set_ylabel('Precio', size=10)
ax.set_xlabel('Distncia al CBD (kms)',size =10)
#plt.show()


ppreg=pd.DataFrame(db[['Regionname','Price']]).groupby(['Regionname']).agg(['mean'])
print('Region/price')
print(ppreg.head())

p=ppreg.style.format(precision=0, na_rep='MISSING', thousands=' ', formatter={('Price', 'mean'): lambda x: '$ {:,.2f}'.format(x)})
print(p)

def highlight_max(s, props=''):
    return np.where(s ==np.nanmax(s.values), props, '')

p.apply(highlight_max, props = 'color:white;background-color:darkblue', axis=0)

p.apply(highlight_max, props='color:white;background-color:pink;', axis=1)\
  .apply(highlight_max, props='color:white;background-color:purple', axis=0)

cm = sns.light_palette('green', as_cmap=True)

ppreg.style.background_gradient(cmap=cm)\
    .format(precision = 0, na_rep = 'MISSING', thousands =' ',
            formatter = {('Price', 'mean'): lambda x: '${:,.2f}'.format(x)})

ppreg.plot.barh(figsize=(4,4), title='Precio promedio de casas por cada Region');
#plt.show()

#max(db.year)
#min(db.year)

yp=pd.DataFrame(db[['year','Price']].groupby(['year']).agg(['mean']))
print(yp)
yp.plot.bar(figsize=(4,4), title='Precio promedio de casa segun año de venta')
#plt.show()

ra=pd.DataFrame(db[['SellerG', 'Price']].groupby('SellerG').agg(['sum']))
print(ra)

ra.columns=['Total_Sale']
ra = ra.nlargest(10,['Total_Sale'])
ra.plot.barh(figsize = (4,4), title ='Top 10 mejores vendedores con mayor venta totales')
#plt.show()

db.boxplot('Price', 'Method', figsize = (6,4), color = 'blue')
plt.title('Precio segun metodo de pago', size= 8)
#plt.show()

plt.figure(figsize = (6,6))
sns.heatmap(db.corr(), annot = True)
plt.title('Matriz de correlaciones', size = 12)
#plt.show()

#Adicionando gráficos interactivos usando plotly
#https://plotly.com/python/reference/layout/

df= db.copy()
df['Regionname'].unique()
#print(df['Regionname'].unique())


#Precio por region
all_regions = df['Price'].values
northern_metropolitan = df['Price'].loc[df['Regionname'] == 'Northern Metropolitan'].values
southern_metropolitan = df['Price'].loc[df['Regionname'] == 'Southern Metropolitan'].values
eastern_metropolitan = df['Price'].loc[df['Regionname'] == 'Eastern Metropolitan'].values
western_metropolitan = df['Price'].loc[df['Regionname'] == 'Western Metropolitan'].values
southeastern_metropolitan = df['Price'].loc[df['Regionname'] == 'South-Eastern Metropolitan'].values
northern_victoria = df['Price'].loc[df['Regionname'] == 'Northern Victoria'].values
eastern_victoria = df['Price'].loc[df['Regionname'] == 'Eastern Victoria'].values
western_victoria = df['Price'].loc[df['Regionname'] == 'Western Victoria'].values
gaussian_distribution = np.log(df['Price'].values)

# Histogramas
overall_price_plot = go.Histogram(
    x=all_regions,
    histnorm='density',
    name='All Regions',
    marker=dict(
        color='#6E6E6E'
    )
)


northern_metropolitan_plot = go.Histogram(
    x=northern_metropolitan,
    histnorm='density',
    name='Northern Metropolitan',
    marker=dict(
        color='#2E9AFE'
    )
)

southern_metropolitan_plot = go.Histogram(
    x=southern_metropolitan,
    histnorm='density',
    name='Southern Metropolitan',
    marker=dict(
        color='#FA5858'
    )
)


eastern_metropolitan_plot = go.Histogram(
    x=eastern_metropolitan,
    histnorm='density',
    name='Eastern Metropolitan',
    marker=dict(
        color='#81F781'
    )
)

western_metropolitan_plot = go.Histogram(
    x=western_metropolitan,
    histnorm='density',
    name='Western Metropolitan',
    marker=dict(
        color='#BE81F7'
    )
)

southeastern_metropolitan_plot = go.Histogram(
    x=southeastern_metropolitan,
    histnorm='density',
    name='SouthEastern Metropolitan',
    marker=dict(
        color='#FE9A2E'
    )
)

northern_victoria_plot = go.Histogram(
    x=northern_victoria,
    histnorm='density',
    name='Northern Victoria',
    marker=dict(
        color='#04B4AE'
    )
)

eastern_victoria_plot = go.Histogram(
    x=eastern_victoria,
    histnorm='density',
    name='Eastern Victoria',
    marker=dict(
        color='#088A08'
    )
)


western_victoria_plot = go.Histogram(
    x=western_victoria,
    histnorm='density',
    name='Western Victoria',
    marker=dict(
        color='#8A0886'
    )
)

gaussian_distribution_plot = go.Histogram(
    x=gaussian_distribution,
    histnorm='probability',
    name='Gaussian Distribution',
    marker=dict(
        color='#002060'
    )
)

fig = tools.make_subplots(rows=6, cols=2, print_grid=False, specs=[[{'colspan': 2}, None], [{}, {}], [{}, {}], [{}, {}], [{}, {}], [{'colspan': 2}, None]],
                         subplot_titles=(
                             'Overall Price Distribution',
                             'Northern Metropolitan',
                             'Southern Metropolitan',
                             'Eastern Metropolitan',
                             'Western Metropolitan',
                             'SouthEastern Metropolitan',
                             'Northern Victoria',
                             'Eastern Victoria',
                             'Western Victoria',
                             'Gaussian Distribution of Price'
                             ))
fig.append_trace(overall_price_plot, 1, 1)
fig.append_trace(northern_metropolitan_plot, 2, 1)
fig.append_trace(southern_metropolitan_plot, 2, 2)
fig.append_trace(eastern_metropolitan_plot, 3, 1)
fig.append_trace(western_metropolitan_plot, 3, 2)
fig.append_trace(southeastern_metropolitan_plot, 4, 1)
fig.append_trace(northern_victoria_plot, 4, 2)
fig.append_trace(eastern_victoria_plot, 5, 1)
fig.append_trace(western_victoria_plot, 5, 2)
fig.append_trace(gaussian_distribution_plot, 6, 1)

fig['layout'].update(showlegend=False, title="Distribución del Precio por Región",
                    height=1200, width=1000)


# Convertir fecha a formato Datetime
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")

# Analizar la estacionalidad por mes (y responder a la pregunta en qué mes hay más demanda)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

total_sales = df['Price'].sum()

#definimos la función para el cálculo de la participación de las ventas.
def month_sales(df, month, sales=total_sales):
    share_month_sales = df['Price'].loc[df['Month'] == month].sum()/sales
    return share_month_sales


#usamos la función y calculamos la participación por mes
january_sales = month_sales(df, 1)
february_sales = month_sales(df, 2)
march_sales = month_sales(df, 3)
april_sales = month_sales(df, 4)
may_sales = month_sales(df, 5)
june_sales = month_sales(df, 6)
july_sales = month_sales(df, 7)
august_sales = month_sales(df, 8)
september_sales = month_sales(df, 9)
october_sales = month_sales(df, 10)
november_sales = month_sales(df, 11)
december_sales = month_sales(df, 12)

month_total_sales = [january_sales, february_sales, march_sales, april_sales,
                     may_sales, june_sales, july_sales, august_sales,
                     september_sales, october_sales, november_sales, december_sales]

labels = ['January', 'February', 'March', 'April',
          'May', 'June', 'July', 'August', 'September',
          'October', 'November', 'December']


colors = ['#ffb4da', '#b4b4ff', '#daffb4', '#fbab60', '#fa8072', '#FA6006',
          '#FDB603', '#639702', '#dacde6', '#faec72', '#9ab973', '#87cefa']

pie_plot = go.Pie(labels=labels, values=month_total_sales,
               hoverinfo='label+percent',
               marker=dict(colors=colors,
                           line=dict(color='#D3D3D3', width=1)))

data = [pie_plot]

layout = go.Layout(
    title="Participación de las ventas por mes"
)

fig = go.Figure(data=data, layout=layout)

# definimos una función de suma de ventas por mes y año.
def month_year_sales(df, month, year):
    double_conditional = df['Price'].loc[(df['Month'] == month) & (df['Year'] == year)].sum()
    return


# Ventas 2016
january_2016 = month_year_sales(df, 1, 2016)
february_2016 = month_year_sales(df, 2, 2016)
march_2016 = month_year_sales(df, 3, 2016)
april_2016 = month_year_sales(df, 4, 2016)
may_2016 = month_year_sales(df, 5, 2016)
june_2016 = month_year_sales(df, 6, 2016)
july_2016 = month_year_sales(df, 7, 2016)
august_2016 = month_year_sales(df, 8, 2016)
september_2016 = month_year_sales(df, 9, 2016)
october_2016 = month_year_sales(df, 10, 2016)
november_2016 = month_year_sales(df, 11, 2016)
december_2016 = month_year_sales(df, 12, 2016)

# Ventas 2017
january_2017 = month_year_sales(df, 1, 2017)
february_2017 = month_year_sales(df, 2, 2017)
march_2017 = month_year_sales(df, 3, 2017)
april_2017 = month_year_sales(df, 4, 2017)
may_2017 = month_year_sales(df, 5, 2017)
june_2017 = month_year_sales(df, 6, 2017)
july_2017 = month_year_sales(df, 7, 2017)
august_2017 = month_year_sales(df, 8, 2017)
september_2017 = month_year_sales(df, 9, 2017)
october_2017 = month_year_sales(df, 10, 2017)
november_2017 = month_year_sales(df, 11, 2017)
december_2017 = month_year_sales(df, 12, 2017)

# Vetas 2018 (hasta Mayo)
january_2018 = month_year_sales(df, 1, 2018)
february_2018 = month_year_sales(df, 2, 2018)
march_2018 = month_year_sales(df, 3, 2018)
april_2018 = month_year_sales(df, 4, 2018)
may_2018 = month_year_sales(df, 5, 2018)


# List of values
lst_2016 = [january_2016, february_2016, march_2016, april_2016,
           may_2016, june_2016, july_2016, august_2016,
           september_2016, october_2016, november_2016, december_2016]

lst_2017 = [january_2017, february_2017, march_2017, april_2017,
           may_2017, june_2017, july_2017, august_2017,
           september_2017, october_2017, november_2017, december_2017]


lst_2018 = [january_2018, february_2018, march_2018, april_2018,
           may_2018]


plot_2016 = go.Scatter(
    x=lst_2016,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2016',
    marker=dict(
        color='rgba(0, 128, 128, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)


plot_2017 = go.Scatter(
    x=lst_2017,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2017',
    marker=dict(
        color='rgba(255, 72, 72, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)

plot_2018 = go.Scatter(
    x=lst_2018,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2018',
    marker=dict(
        color='rgba(72, 255, 72, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)

data = [plot_2016, plot_2017, plot_2018]

layout = go.Layout(
    title="Ventas por mes y por año <br> (2016, 2017, 2018)",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(102, 102, 102)',
        titlefont=dict(
            color='rgb(204, 204, 204)'
        ),
        tickfont=dict(
            color='rgb(102, 102, 102)',
        ),
        #autotick=False,
        #tick0=0.75,
        ticks='outside',
        tickcolor='rgb(102, 102, 102)',
    ),
    margin=dict(
        l=140,
        r=40,
        b=50,
        t=80
    ),
    legend=dict(
        font=dict(
            size=10,
        ),
        yanchor='top',
        xanchor='left',
    ),
    width=800,
    height=600,
    paper_bgcolor='rgb(255, 255, 224)',
    plot_bgcolor='rgb(255, 255, 250)',
    hovermode='x unified', # puede ser x, x unified,closest
)

fig = go.Figure(data=data, layout=layout)

# definir función para el precio promedio de venta por mes
def avg_price_sold(df, month, year):
    avg_p = round(np.mean(df['Price'].loc[(df['Month'] == month) & (df['Year'] == year)].values), 2)
    return avg_p


#usamos la función y generamos los promedios por mes para cada año
# 2016 - meses
jan_2016 = avg_price_sold(df, 1, 2016)
feb_2016 = avg_price_sold(df, 2, 2016)
mar_2016 = avg_price_sold(df, 3, 2016)
apr_2016 = avg_price_sold(df, 4, 2016)
may_2016 = avg_price_sold(df, 5, 2016)
june_2016 = avg_price_sold(df ,6, 2016)
july_2016 = avg_price_sold(df, 7, 2016)
aug_2016 = avg_price_sold(df, 8, 2016)
sep_2016 = avg_price_sold(df, 9, 2016)
oct_2016 = avg_price_sold(df, 10, 2016)
nov_2016 = avg_price_sold(df, 11, 2016)
dec_2016 = avg_price_sold(df, 12, 2016)

# 2017 - meses
jan_2017 = avg_price_sold(df, 1, 2017)
feb_2017 = avg_price_sold(df, 2, 2017)
mar_2017 = avg_price_sold(df, 3, 2017)
apr_2017 = avg_price_sold(df, 4, 2017)
may_2017 = avg_price_sold(df, 5, 2017)
june_2017 = avg_price_sold(df ,6, 2017)
july_2017 = avg_price_sold(df, 7, 2017)
aug_2017 = avg_price_sold(df, 8, 2017)
sep_2017 = avg_price_sold(df, 9, 2017)
oct_2017 = avg_price_sold(df, 10, 2017)
nov_2017 = avg_price_sold(df, 11, 2017)
dec_2017 = avg_price_sold(df, 12, 2017)

# 2018 - meses
jan_2018 = avg_price_sold(df, 1, 2018)
feb_2018 = avg_price_sold(df, 2, 2018)
mar_2018 = avg_price_sold(df, 3, 2018)
apr_2018 = avg_price_sold(df, 4, 2018)
may_2018 = avg_price_sold(df, 5, 2018)

# Listas para cada año
# 2016
lst_2016_avg = [jan_2016, feb_2016, mar_2016, apr_2016, may_2016, june_2016, july_2016, aug_2016, sep_2016,
               oct_2016, nov_2016, dec_2016]
# 2017
lst_2017_avg = [jan_2017, feb_2017, mar_2017, apr_2017, may_2017, june_2017, july_2017, aug_2017, sep_2017,
               oct_2017, nov_2017, dec_2017]
# 2018
lst_2018_avg = [jan_2018, feb_2018, mar_2018, apr_2018, may_2018]

# hacemos algunos reemplazos antes de graficar, todos los nan por ceros
lst_2016_avg[2] = 0
lst_2017_avg[0] = 0
lst_2018_avg[0] = jan_2018


# Grafica radar para los tres años - distribución de las ventas mensuales.
month_labels = ['January', 'February', 'March', 'April',
                'May', 'June', 'July', 'August', 'September',
                'October', 'November', 'December']

data = [
    go.Scatterpolar(
        mode='lines+markers',
        r = lst_2016_avg,
        theta = month_labels,
        fill = 'toself',
        name="2016",
        line=dict(
            color="rgba(0, 128, 128, 0.95)"
        ),
        marker=dict(
            color="rgba(0, 74, 147, 1)",
            symbol="square",
            size=8
        ),
        subplot = "polar"
    ),
    go.Scatterpolar(
        mode='lines+markers',
        r = lst_2017_avg,
        theta = month_labels,
        fill = 'toself',
        name="2017",
        line=dict(
            color="rgba(255, 72, 72, 0.95)"
        ),
        marker=dict(
            color="rgba(219, 0, 0, 1)",
            symbol="square",
            size=8
        ),
        subplot = "polar2"
    ),
    go.Scatterpolar(
        mode='lines+markers',
        r = lst_2018_avg,
        theta = month_labels,
        fill = 'toself',
        name="2018",
        line=dict(
            color="rgba(72, 255, 72, 0.95)"
        ),
        marker=dict(
            color="rgba(0, 147, 74, 1)",
            symbol="square",
            size=8
        ),
        subplot = "polar3"
    )
]

layout = go.Layout(
    title="Precio promedio de casas <br> (Distribución por mes y año)",
    showlegend = False,
     paper_bgcolor = "rgb(255, 255, 224)",
    polar = dict(
      domain = dict(
        x = [0,0.3],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 10
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 10
        ),
        rotation = 90,
        direction = "counterclockwise"
      )
    ),
    polar2 = dict(
      domain = dict(
        x = [0.35,0.65],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 10
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 10
        ),
        rotation = 85,
        direction = "clockwise"
      ),
    ),
    polar3 = dict(
      domain = dict(
        x = [0.7, 1],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 10
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size =10
        ),
        rotation = 90,
        direction = "clockwise"
      ),
    ))

fig = go.Figure(data=data, layout=layout)
#iplot(fig, filename='polar/directions')

df['Month'].value_counts()

df['Season'] = np.nan  #generamos una nueva variable y le asignamos cero valores

lst = [df]
lst

# Adicionamos valores a la variable Season
for column in lst:
    column.loc[(column['Month'] > 2) & (column['Month'] <= 5), 'Season'] = 'Spring'
    column.loc[(column['Month'] > 5) & (column['Month'] <= 8), 'Season'] = 'Summer'
    column.loc[(column['Month'] > 8) & (column['Month'] <= 11), 'Season'] = 'Autumn'
    column.loc[column['Month'] <= 2, 'Season'] = 'Winter'
    column.loc[column['Month'] == 12, 'Season'] = 'Winter'

df['Season'].value_counts()  # revisamos cuantos registros tiene cada estación
#graficamos las distribución del precio de los tipos de cada por estación, usando un gráfico violin
fig = {
    "data": [
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Spring' ],
            "y": df['Price'] [ df['Season'] == 'Spring' ],
            "legendgroup": 'Spring',
            "scalegroup": 'Spring',
            "name": 'Spring',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#6cff6c'
            }
        },
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Summer' ],
            "y": df['Price'] [ df['Season'] == 'Summer' ],
            "legendgroup": 'Summer',
            "scalegroup": 'Summer',
            "name": 'Summer',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#ff6961'
            }
        },
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Autumn' ],
            "y": df['Price'] [ df['Season'] == 'Autumn' ],
            "legendgroup": 'Autumn',
            "scalegroup": 'Autumn',
            "name": 'Autumn',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#9a5755'
            }
        },
        {
            "type": 'violin',
            "x": df['Type'] [ df['Season'] == 'Winter' ],
            "y": df['Price'] [ df['Season'] == 'Winter' ],
            "legendgroup": 'Winter',
            "scalegroup": 'Winter',
            "name": 'Winter',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#70c8cb'
            }
        },
    ],
    "layout" : {
        "title": "Distribución del precio según tipo de casa<br> <sub> Midiendo según la estacionalidad</sub>",
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}


#iplot(fig, filename = 'violin/grouped', validate = False)


# deinimos una función para el % de ventas
def region_sales_percentage(df, region, sales=total_sales):
    sales_percentage = (df['Price'].loc[df['Regionname'] == region].sum()/sales) * 100
    return sales_percentage


# %de ventas por región
northernmet_salesper = region_sales_percentage(df, region='Northern Metropolitan')
westernmet_salesper = region_sales_percentage(df, region='Western Metropolitan')
southernmet_salesper = region_sales_percentage(df, region='Southern Metropolitan')
easternmet_salesper = region_sales_percentage(df, region='Eastern Metropolitan')
south_easternmet_salesper = region_sales_percentage(df, region='South-Eastern Metropolitan')
northernvic_salesper = region_sales_percentage(df, region='Northern Victoria')
westernvic_salesper = region_sales_percentage(df, region='Western Victoria')
easternvic_salesper = region_sales_percentage(df, region='Eastern Victoria')

# Suma total de las ventas por región
nothernmet_total_sales = df['Price'].loc[df['Regionname'] == 'Northern Metropolitan'].sum()
westernmet_total_sales = df['Price'].loc[df['Regionname'] == 'Western Metropolitan'].sum()
southernmet_total_sales = df['Price'].loc[df['Regionname'] == 'Southern Metropolitan'].sum()
easternmet_total_sales = df['Price'].loc[df['Regionname'] == 'Eastern Metropolitan'].sum()
south_easternmet_total_sales = df['Price'].loc[df['Regionname'] == 'South-Eastern Metropolitan'].sum()
northernvic_total_sales = df['Price'].loc[df['Regionname'] == 'Northern Victoria'].sum()
westernvic_total_sales = df['Price'].loc[df['Regionname'] == 'Western Victoria'].sum()
easternvic_total_sales = df['Price'].loc[df['Regionname'] == 'Eastern Victoria'].sum()

labels = ['Northern <br> Metropolitan', 'Western <br> Metropolitan', 'Southern <br> Metropolitan',
          'Eastern <br> Metropolitan',
          'South-Eastern <br> Metropolitan', 'Northern <br> Victoria', 'Western <br> Victoria', 'Eastern <br> Victoria']

salesper_data = [northernmet_salesper, westernmet_salesper, southernmet_salesper, easternmet_salesper,
                 south_easternmet_salesper, northernvic_salesper, westernvic_salesper, easternvic_salesper]

total_sales_data = [nothernmet_total_sales, westernmet_total_sales, southernmet_total_sales, easternmet_total_sales,
                    south_easternmet_total_sales, northernvic_total_sales, westernvic_total_sales,
                    easternvic_total_sales]

sales_percent_plot = go.Bar(
    x=salesper_data,
    y=labels,
    marker=dict(
        color='rgba(152, 251, 152, 0.6)',
        line=dict(
            color='rgba(12, 218, 12, 1)',
            width=1),
    ),
    name='%de Ventas de casas por región',
    orientation='h'
)

total_sales_plot = go.Scatter(
    x=total_sales_data,
    y=labels,
    mode='markers',
    marker=dict(
        color='rgb(34, 178, 178)',
        size=8),
    name='Total de Ventas por Región / (en dólares australianos ($AUD))'
)

layout = go.Layout(
    title='Total de ventas en Melbourne <br> Mercado inmbiliario',
    yaxis1=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        domain=[0, 0.85],
        tickangle=360
        # tickcolor=dict(
        # color='antiquewhite'

    ),
    yaxis2=dict(
        showgrid=True,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
        tickangle=360
    ),
    xaxis1=dict(
        zeroline=True,
        showline=True,
        showticklabels=False,
        showgrid=True,
        domain=[0, 0.42],
    ),
    xaxis2=dict(
        zeroline=True,
        showline=True,
        showticklabels=False,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=25000,
    ),
    legend=dict(
        x=0.029,
        y=1.038,
        font=dict(
            size=10,
        ),
    ),
    margin=dict(
        l=100,
        r=220,
        t=70,
        b=70,
    ),
    paper_bgcolor='rgb(255, 255, 224)',
    plot_bgcolor='rgb(255, 255, 246)',
)

y_s = np.round(salesper_data, decimals=2)
y_nw = np.rint(total_sales_data)

annotations = []

# Adicionando las etiquetas
for ydn, yd, xd in zip(y_nw, y_s, labels):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn * 2,
                            text='{:,}'.format(ydn) + 'M',
                            ax=50,
                            ay=20,
                            xanchor='left',
                            yanchor='middle',
                            font=dict(family='Arial', size=12,
                                      color='rgb(17, 87, 87)'),
                            showarrow=False,
                            arrowhead=3))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 6,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(7, 143, 7)'),
                            showarrow=False))

layout['annotations'] = annotations

# creando los dos subplots
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(sales_percent_plot, 1, 1)
fig.append_trace(total_sales_plot, 1, 2)

fig['layout'].update(layout)
#iplot(fig, filename='oecd-networth-saving-bar-line')