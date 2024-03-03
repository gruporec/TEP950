import pandas as pd
import os

path = 'bases de datos originales/anuales/'
path_out = 'bases de datos feather/raw/'
files = [file for file in os.listdir(path)]

for file in files:
     if not os.path.exists(path_out+os.path.splitext(file)[0]+'/'):
          os.makedirs(path_out+os.path.splitext(file)[0]+'/')
          xls = pd.ExcelFile(path+file)
          df1 = pd.read_excel(xls, 0)
          df2 = pd.read_excel(xls, 1)
          df3 = pd.read_excel(xls, 2)
          df4 = pd.read_excel(xls, 3)
          df5 = pd.read_excel(xls, 4,header=[0,1])
          #df['column'] = df['column'].astype(str)
          #df['column'] = df['column'].replace('old charecter causing error','new charecter').astype(str)
          #df1=df1.set_index('Fecha')
               
          df2.columns=(df2.columns+' '+df2.iloc[0,:].astype(str).replace('NaT', '')).str.strip(to_strip=None)
          df2=df2.drop(0)
          df2=df2.reset_index()
          #df2=df2.set_index('Fecha')

          df3aux2=pd.DataFrame()
          for i in range(int(len(df3.columns)/2)):
               df3aux1=df3.iloc[:,2*i:2*i+2]
               df3aux1.columns=['Fecha',df3aux1.columns[1]+' '+df3aux1.iloc[0,1]]
               df3aux1=df3aux1.drop(0)
               df3aux1=df3aux1.dropna()
               df3aux1=df3aux1.set_index('Fecha')
               df3aux1.index = pd.to_datetime(df3aux1.index)
               df3aux1.iloc[:,0]=pd.to_numeric(df3aux1.iloc[:,0])
               df3aux1=df3aux1.resample('T').interpolate()
               df3aux2=pd.concat([df3aux2,df3aux1],axis=1)
          df3=df3aux2
          df3=df3.reset_index()

          df4.columns.values[0]='Fecha'
          #df4=df4.set_index('Fecha')

          df5=df5.set_index(df5.columns[0])
          df5.index.names=['Fecha']
          df5.reset_index()

          df1 = df1.replace({'NAN': None})
          df2 = df2.replace({'NAN': None})
          df3 = df3.replace({'NAN': None})
          df4 = df4.replace({'NAN': None})
          df5 = df5.replace({'NAN': None})
          df1.to_feather(path_out+os.path.splitext(file)[0]+'/MeteoDiarios'+'.feather')
          df2.to_feather(path_out+os.path.splitext(file)[0]+'/TDV'+'.feather')
          df3.to_feather(path_out+os.path.splitext(file)[0]+'/LTP'+'.feather')
          df4.to_feather(path_out+os.path.splitext(file)[0]+'/MeteoSubH'+'.feather')
          df5['ZIMS'].reset_index().to_feather(path_out+os.path.splitext(file)[0]+'/Estados ZIM'+'.feather')
          df5['DENDROMETROS'].reset_index().to_feather(path_out+os.path.splitext(file)[0]+'/Estados Dendrometros'+'.feather')
          df5['ESTADO HIDRICO PARCELAS'].reset_index().to_feather(path_out+os.path.splitext(file)[0]+'/Estados Parcelas'+'.feather')