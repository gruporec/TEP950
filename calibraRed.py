from re import A
import pandas as pd
import scipy.optimize as op

def calculaEstado(x,min,max):
    estado=2
    if x<min:
        estado=3
    if x>max:
        estado=1
    return estado

def error(csv,primsen,len,est,sens):
    df=pd.read_csv(csv)
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    df=df.set_index("Fecha")

    dfe1=est.iloc[:,sens:len*2:2]
    dfe1.dropna(inplace=True)
    dfs=df.iloc[:,primsen:primsen+len]
    dfs.dropna(inplace=True)
    dfs1=dfs
    dfs1.columns=dfe1.columns
    ret=abs(dfs1-dfe1).mean()

    return ret

def calculaEstSum(design_variables):#a,an,m,r,mn
    a=int(design_variables[0])
    an=int(design_variables[1])
    m=int(design_variables[2])
    r=int(design_variables[3])
    mn=int(design_variables[4])
    dfT = pd.read_csv("rawMinutales.csv",na_values='.') 
    dfT.iloc[:,0]=pd.to_datetime(dfT.iloc[:,0]) # Fecha como datetime
    dfT=dfT.drop_duplicates(subset="Fecha")
    dfT.dropna(subset = ["Fecha"], inplace=True)
    dfT=dfT.set_index("Fecha")

    dfd = pd.read_csv("rawDiarios.csv",na_values='.') 
    dfd.iloc[:,0]=pd.to_datetime(dfd.iloc[:,0]) # Fecha como datetime
    dfd=dfd.drop_duplicates(subset="Fecha")
    dfd.dropna(subset = ["Fecha"], inplace=True)
    dfd=dfd.set_index("Fecha")

    lpt=dfT.iloc[:,12:36]
    lpt0=lpt.resample('1D').first()
    lptmax=lpt.resample('1D').max()-lpt0
    lpti=lpt.resample('1D').sum()/(60*24)-lpt0
    lptin=lpti/(lptmax)
    #lpti=lpti.add_prefix("Area ")
    #lptin=lptin.add_prefix("Area norm ")
    #lptmax=lptmax.add_prefix("Maximo diario ")

    #dfdp=pd.merge(dfd,lpti,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,lptin,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,lptmax,how="outer",on="Fecha")

    #estArea=lpti.rolling(14).mean().applymap(calculaEstado,min=0,max=15)
    #estAreaN=lptin.rolling(14).mean().applymap(calculaEstado,min=-0.4,max=0.2)
    #estMax=lptmax.rolling(14).mean().applymap(calculaEstado,min=30,max=55)

    estInd=lpti*a+lptin*an+lptmax*m
    est=estInd.rolling(r).mean().applymap(calculaEstado,min=mn,max=10)

    #dfdp=pd.merge(dfdp,estArea,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,estAreaN,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,estMax,how="outer",on="Fecha")
    ret=error("estadisticosDiarios.csv",7,12,est,0)
    ret=ret*ret
    ret=ret.sum()
    return ret
def calculaEst(design_variables,sens):#a,an,m,r,mn,mx
    a=float(design_variables[0])
    an=float(design_variables[1])
    m=float(design_variables[2])
    r=int(design_variables[3])
    mn=float(design_variables[4])
    dfT = pd.read_csv("rawMinutales.csv",na_values='.') 
    dfT.iloc[:,0]=pd.to_datetime(dfT.iloc[:,0]) # Fecha como datetime
    dfT=dfT.drop_duplicates(subset="Fecha")
    dfT.dropna(subset = ["Fecha"], inplace=True)
    dfT=dfT.set_index("Fecha")

    dfd = pd.read_csv("rawDiarios.csv",na_values='.') 
    dfd.iloc[:,0]=pd.to_datetime(dfd.iloc[:,0]) # Fecha como datetime
    dfd=dfd.drop_duplicates(subset="Fecha")
    dfd.dropna(subset = ["Fecha"], inplace=True)
    dfd=dfd.set_index("Fecha")

    lpt=dfT.iloc[:,12:36]
    lpt0=lpt.resample('1D').first()
    lptmax=lpt.resample('1D').max()-lpt0
    lpti=lpt.resample('1D').sum()/(60*24)-lpt0
    lptin=lpti/(lptmax)
    #lpti=lpti.add_prefix("Area ")
    #lptin=lptin.add_prefix("Area norm ")
    #lptmax=lptmax.add_prefix("Maximo diario ")

    #dfdp=pd.merge(dfd,lpti,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,lptin,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,lptmax,how="outer",on="Fecha")

    #estArea=lpti.rolling(14).mean().applymap(calculaEstado,min=0,max=15)
    #estAreaN=lptin.rolling(14).mean().applymap(calculaEstado,min=-0.4,max=0.2)
    #estMax=lptmax.rolling(14).mean().applymap(calculaEstado,min=30,max=55)

    estInd=lpti*a+lptin*an+lptmax*m
    est=estInd.rolling(r).mean().applymap(calculaEstado,min=mn,max=10)

    #dfdp=pd.merge(dfdp,estArea,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,estAreaN,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,estMax,how="outer",on="Fecha")
    ret=error("estadisticosDiarios.csv",7,12,est,sens)
    return est
    #dfdp.to_csv("estadisticosDiarios.csv", index=True)
def calculaEstim(design_variables,sens):#a,an,m,r,mn,mx
    a=float(design_variables[0])
    an=float(design_variables[1])
    m=float(design_variables[2])
    r=int(design_variables[3])
    mn=float(design_variables[4])
    dfT = pd.read_csv("rawMinutales.csv",na_values='.') 
    dfT.iloc[:,0]=pd.to_datetime(dfT.iloc[:,0]) # Fecha como datetime
    dfT=dfT.drop_duplicates(subset="Fecha")
    dfT.dropna(subset = ["Fecha"], inplace=True)
    dfT=dfT.set_index("Fecha")

    dfd = pd.read_csv("rawDiarios.csv",na_values='.') 
    dfd.iloc[:,0]=pd.to_datetime(dfd.iloc[:,0]) # Fecha como datetime
    dfd=dfd.drop_duplicates(subset="Fecha")
    dfd.dropna(subset = ["Fecha"], inplace=True)
    dfd=dfd.set_index("Fecha")

    lpt=dfT.iloc[:,12:36]
    lpt0=lpt.resample('1D').first()
    lptmax=lpt.resample('1D').max()-lpt0
    lpti=lpt.resample('1D').sum()/(60*24)-lpt0
    lptin=lpti/(lptmax)
    #lpti=lpti.add_prefix("Area ")
    #lptin=lptin.add_prefix("Area norm ")
    #lptmax=lptmax.add_prefix("Maximo diario ")

    #dfdp=pd.merge(dfd,lpti,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,lptin,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,lptmax,how="outer",on="Fecha")

    #estArea=lpti.rolling(14).mean().applymap(calculaEstado,min=0,max=15)
    #estAreaN=lptin.rolling(14).mean().applymap(calculaEstado,min=-0.4,max=0.2)
    #estMax=lptmax.rolling(14).mean().applymap(calculaEstado,min=30,max=55)

    estInd=3-((lpti*a+lptin*an+lptmax*m)/10)
    estInd=estInd.rolling(r).mean()

    #dfdp=pd.merge(dfdp,estArea,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,estAreaN,how="outer",on="Fecha")
    #dfdp=pd.merge(dfdp,estMax,how="outer",on="Fecha")
    return estInd

x0=(4, 8, 0, 3, 0)
est=calculaEst(x0,0)
#print(calculaEst(x0,1))
est=est.add_prefix("Estado estimado ")
estim=calculaEstim(x0,0)
estim=estim.add_prefix("Estimador de estado ")
ret=error("estadisticosDiarios.csv",7,12,est,0)
print(ret)
ret=error("estadisticosDiarios.csv",7,12,est,1)
print(ret)
df=pd.read_csv("estadisticosDiarios.csv")
df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
df=df.drop_duplicates(subset="Fecha")
df.dropna(subset = ["Fecha"], inplace=True)
df=df.set_index("Fecha")
df=pd.merge(df,est,how="outer",on="Fecha")
df=pd.merge(df,estim,how="outer",on="Fecha")
df.to_csv("estadisticosDiariosConEst.csv", index=True)

#bestval=op.minimize(calculaEstSum,x0,method='nelder-mead',options={'return_all':True,'disp':True})
#print(bestval)

# minerr=100
# bestfit=(-10,-10,-10,-10,-10)
# for a in range(-10,10,2): #a,an,m,r,mn,mx
#     for an in range(-10,10,2):
#         for m in range(-10,10,2):
#             for r in range(0,21,3):
#                 for mn in range(-10,10,2):
#                     x0=(a,an,m,r,mn)
#                     err=calculaEst(x0,0)
#                     errsq=err*err
#                     sumerr=errsq.sum()
#                     if sumerr<minerr:
#                         bestfit=x0
#                         minerr=sumerr
#                         print(sumerr)
#                         print(bestfit)
#                 percent=((((r/21)+m+10)/20+an+10)/20+a+10)*10/2
#                 print('Evaluado: '+str(percent)+'%')
# print('El mejor resultado ha sido:'+str(sumerr)+' con los valores:'+str(bestfit))

#El mejor resultado ha sido:6.584859404536861 con los valores:(4, 8, 0, 3, 0)

