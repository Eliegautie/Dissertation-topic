#!/usr/bin/env python
# coding: utf-8

# **Modèle de prévision du benchmark européen du gaz TTF**

# In[98]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt

import seaborn as sns       

from numpy                 import array
from sklearn               import metrics
from sklearn.preprocessing import StandardScaler  
from tensorflow.keras.models          import Sequential 
from tensorflow.keras.layers          import LSTM
from tensorflow.keras.layers import Dense, Dropout


# **1 : Type de données collectées** 
# 
# *Données climatiques :*
# 
# le gaz est utilisé majorité en Europe pour la production d'électricité et de chaleur. Les variations de température ont donc un impact sur la demande de gaz naturel. Plus il fait froid, plus la demande en gaz naturel augmente, d'où l'importance de remplir au maximum les stockages de gaz naturel en été, lorsque les prix sur les marchés sont généralement moins élevés. 
# 
# *Données de stockage :* 
# 
# Lorsque les niveaux de stockage sont bas, cela crée une tension sur les marchés. Pour garantir la sécurité énergétique, les importations de gaz par pipeline ou LNG augmentent. 
# 
# *Données d'approvisionnement en gaz :*
# 
# J'ai pris en compte le gaz acheminé par pipeline et par LNG en Europe. Depuis la guerre en Ukraine, La Russie, principal fournisseur de gaz en Europe, a fermé petit à petit les chemins d'approvisionnement. La grande dépendance de certains états européens de l'est (notamment l'Allemagne) en gaz russe a créé une hausse des prix fulgurante en 2022. 
# 
# *Données de marché :*
# 
# Le STOXX 600 est un indice boursier composite pour l'Europe qui reflète le niveau d'activité des marchés financiers européens et illustre la corrélation entre le prix à terme du gaz naturel en tant que produit de base et le marché.

# In[99]:


ttf= pd.read_csv(r'C:\Users\Nomade05\Other\Desktop\memoire_elie\TTF\TTF_price.csv')
stoxx = pd.read_csv(r'C:\Users\Nomade05\Other\Desktop\memoire_elie\STOXX600\stoxx_1.csv')
temperature = pd.read_csv(r'C:\Users\Nomade05\Other\Desktop\temp\temp.csv')
ng = pd.read_csv(r'C:\Users\Nomade05\Other\Desktop\memoire_elie\NG_volume\ngvolume.csv')
inventory = pd.read_csv(r'C:\Users\Nomade05\Other\Desktop\memoire_elie\Inventory\inventory.csv')


# 
# 
# **TTF**

# In[100]:


ttf


# **STOXX 600**

# In[101]:


stoxx


# **Natural gas volume**

# In[102]:


ng


# **Temperatures**

# In[103]:


temperature.rename(columns={'    DATE': 'Date'}, inplace=True)
temperature


# **Gas in storage**

# In[104]:


inventory.rename(columns={'Gas Day Start': 'Date'}, inplace=True)
inventory


# **2 : Corrélation entre stockage et températures**

# In[105]:


fig, ax1 = plt.subplots(figsize=(12,4))

ax1.plot(temperature['Date'], temperature['AVG.(celsius)'], 'r-', label='Temperature')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature', color='r')
ax1.tick_params(axis='y', labelcolor='r')
 

ax1.set_xlabel('')
ax1.xaxis.set_ticks([])

# Créer un second axe Y
ax2 = ax1.twinx()
ax2.plot(inventory['Date'], inventory['Gas in storage (TWh)'], 'b-', label='Gas in storage (TWh)')
ax2.set_ylabel('Gas in storage (TWh)', color='b')
ax2.tick_params(axis='y', labelcolor='b')


# Le graphe ci-dessus montre la corrélation positive entre le niveau de gaz dans les stockages et les températures. Lorsque les températures augmentent, le niveau de gaz dans les stockages augmente avec un certain laps de temps. C'est durant la période hivernale que le gaz est majoritairement utilisé. Les acteurs remplissent les stockages en été, lorsque les prix sur les marchés européens sont bas. On peut remarquer le niveau de stockage particulièrement bas sur l'année 2021-2022 (4e pic), du notamment à la réduction des livraisons de gaz russe durant l'été 2021 et à l'arrêt quasi-total de ces livraisons après l'invasion de l'Ukraine en 2022.

# **3 : Corrélation entre le cours du TTF et du STOXX600**

# In[106]:


stoxx = stoxx.replace([np.inf, -np.inf], np.nan).dropna()


# In[107]:


fig, ax1 = plt.subplots(figsize=(12,4))
ax1.plot(stoxx['stoxx600 close price'], 'r-', label='Stoxx 600')
ax1.set_ylabel('Stoxx 600', color='r')
ax1.tick_params(axis='y', labelcolor='r')

ax1.set_xlabel('')
ax1.xaxis.set_ticks([])

ax2 = ax1.twinx()
ax2.plot(ttf['TTF open price'], 'b-', label='TTF')
ax2.set_ylabel('TTF (EUR/MWh)', color='b')
ax2.tick_params(axis='y', labelcolor='b')


# 
# **4 : Principaux fournisseurs de gaz en Europe**

# In[108]:


ng.tail(3)


# In[109]:


fig=plt.figure(figsize=(12,6))
df_ng = pd.DataFrame([
    ['Netherlands', 434564742.5], ['Denmark', 247261236],['Norway ', 2491303989],  ['Libya', 97790614], ['Algeria', 968246178], ['Romania', 250994763.9],
    ['Russia ', 792325800],  ['Azerbaijan', 349131215 ], ['LNG', 1273458096]],columns=['country', 'supply'])
plt.pie(df_ng['supply'], labels=df_ng['country'], autopct='%1.2f%%')
plt.title('Natural Gas Supply to EU, 10/02/2023')
plt.show()


# Comme le montre le graphe ci-dessus, 36 % du gaz naturel européen provient de Norvège, 18 % de GNL (gaz naturel liquéfié transporté par voie maritime, principalement en provenance des Etats-Unis et du Qatar) et 14 % d'Algérie.

# In[110]:


df0 = pd.merge(ng, inventory,on="Date",how="outer")
df1 = pd.merge(df0, temperature,on='Date',how="outer")
df2 = pd.merge(df1, stoxx,on='Date',how="outer")
df = pd.merge(df2, ttf,on='Date',how="outer")


# In[111]:


df.isnull().sum()


# In[112]:


df = df.fillna(method='ffill')


# In[113]:


df.columns
df = df.drop('cyprus',axis=1)


# In[114]:


df.isnull().sum()


# **5 : matrice de corrélation**

# In[115]:


correlation_matrix = df.corr().round(2)
plt.figure(figsize=(32,16))
sns.heatmap(data=correlation_matrix, annot = True)
plt.show()


# As you can see from the graph above :
# - The correlation between TTF gas price and stoxx600 is 0.5. 
# - The correlation with gas imports into Azerbaidjan is 0.78.  
# - Positive correlation bewteen Norway and Azerbaidjan. 
# - Negative correlation with Russia and Lybian exports

# In[116]:


df.info()


# In[117]:


time_df = pd.to_datetime(df['Date'])
time_df.head(4)


# In[118]:


df_input=df[[
       'Azerbaïdjan', 'Russia ', 'Norway ', 'Netherland ', 'Libya ', 'LNG ',
       'stoxx600 open price', 'stoxx600 high price', 'stoxx600 low price', 'TTF open price']]
df_input.head(4)


# In[119]:


ss = StandardScaler()
df_ss = ss.fit_transform(df_input)


# In[120]:


Xtrain = []
Ytrain = []

future = 1   
past = 7 


# In[121]:


for i in range(past, len(df_ss) - future +1):
    Xtrain.append(df_ss[i - past:i, 0:df_input.shape[1]])
    Ytrain.append(df_ss[i + future - 1:i + future, 9])


# In[122]:


Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)


# In[123]:


Xtrain.shape


# In[124]:


Ytrain.shape


# In[125]:


model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=False)) #True
#model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.15))
model.add(Dense(Ytrain.shape[1]))


# In[126]:


model.compile(optimizer='adam', loss='mse')
model.summary()


# **Réseau LSTM (long short-term memory)**
# 
# Un réseau LSTM est un réseau de neurones récurrent (RNN, Recurrent Neural Network) qui traite les données d’entrée en effectuant une boucle à chaque pas de temps tout en mettant à jour l’état du RNN. L’état du RNN contient des informations mémorisées sur tous les pas de temps précédents. 

# In[127]:


history = model.fit(Xtrain, Ytrain, epochs=100, batch_size=128, validation_split=0.1, verbose=1)


# In[128]:


plt.plot(history.history['loss']    , label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss ')
plt.legend()


# In[129]:


past_n = 1201
predict_days_n = 1200  #let us predict past 15 days

predict_dates = pd.date_range(list(time_df)[-past_n], periods= predict_days_n).tolist()
print(predict_dates)


# In[130]:


prediction = model.predict(Xtrain[-predict_days_n:]) 


# In[131]:


prediction_copies = np.repeat(prediction, df_input.shape[1], axis=-1)
Y_prediction_future = ss.inverse_transform(prediction_copies)[:,9]


# In[132]:


forecast_dates = []
for time_i in predict_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'TTF open Price':Y_prediction_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


# In[133]:


df_forecast.columns


# In[134]:


print(df_forecast.columns)


# In[135]:


df_forecast.columns = df_forecast.columns.str.strip()


# In[136]:


df.columns


# In[137]:


df_forecast.columns


# In[139]:


original = df[['Date']]
original = original.loc[original['Date'] >= '2018-01-31'].copy()
original['Date']=pd.to_datetime(original['Date'])
sns.lineplot(x=original['Date'], y=df[ 'TTF open price'])
sns.lineplot(x=df_forecast['Date'], y=df_forecast[ 'TTF open Price'])

