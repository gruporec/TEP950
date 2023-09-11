import pandas as pd
from pandas.core.tools.datetimes import to_datetime

def calculaEstado(x,min,max):
    estado=2
    if x<min:
        estado=1
    if x>max:
        estado=3
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

def errorEstMed(csv,primsen,est,fini,fend):
    df=pd.read_csv(csv)
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    df=df.set_index("Fecha")
    df=df.loc[fini:fend]
    length=int(len(est.columns)/2)

    est.loc[fini:fend]
    dfe1=est.iloc[:,0:length*2:2]
    dfe2=est.iloc[:,1:length*2:2]
    dfe1.dropna(inplace=True)
    dfe2.dropna(inplace=True)
    dfe2.columns=dfe1.columns
    dfe=(dfe1+dfe2)/2
    dfs=df.iloc[:,primsen:primsen+length]
    dfs.dropna(inplace=True)
    dfs.columns=dfe.columns
    ret=abs(dfs-dfe).mean()

    return ret

def estMed(est):
    length=int(len(est.columns)/2)
    dfe1=est.iloc[:,0:length*2:2]
    dfe2=est.iloc[:,1:length*2:2]
    dfe1.dropna(inplace=True)
    dfe2.dropna(inplace=True)
    dfe2.columns=dfe1.columns
    dfe=(dfe1+dfe2)/2

    return dfe

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

    estInd=lpti*a+lptin*an+lptmax*m+mn
    est=estInd.rolling(r).mean().applymap(calculaEstado,min=2,max=3)

    ret=error("estadisticosDiarios.csv",7,12,est,0)
    ret=ret*ret
    ret=ret.sum()
    return ret

def calculaEst(design_variables):#,fini,fend):#a,an,m,pre,mn,mx=10

    a=float(design_variables[0])
    an=float(design_variables[1])
    m=float(design_variables[2])
    pre=float(design_variables[3])
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

    deltaEstInd=lpti*a+lptin*an+lptmax*m+mn
    estInd=deltaEstInd
    for i in range(1, len(estInd)):
        estInd.iloc[i] = estInd.iloc[i-1] * pre + deltaEstInd.iloc[i]*(1-pre)
    est=deltaEstInd.applymap(calculaEstado,min=2,max=3)

    #ret=errorEstMed("estadisticosDiarios.csv",7,12,est,fini,fend)

    return est
    #dfdp.to_csv("estadisticosDiarios.csv", index=True)

def calculaEstim(design_variables):#a,an,m,pre,mn

    a=float(design_variables[0])
    an=float(design_variables[1])
    m=float(design_variables[2])
    pre=float(design_variables[3])
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

    deltaEstInd=lpti*a+lptin*an+lptmax*m+mn
    estInd=deltaEstInd
    for i in range(1, len(estInd)):
        estInd.iloc[i] = estInd.iloc[i-1] * pre + deltaEstInd.iloc[i]*(1-pre)

    return estInd

bestfit=(-0.65, 1.7, 0.0, 0.3, 1.25) #Mueve el centro al bestfit en cada iteración. si ya se encuentra alli, reduce el paso a la mitad.
x0=bestfit
x1=x0
step=1
acenter=bestfit[0]
ancenter=bestfit[1]
mcenter=bestfit[2]
rcenter=bestfit[3]
mncenter=bestfit[4]
minstep=0.01
minerr=100

while step>minstep:
    for a in range(-1,2,1): 
        a=a*step
        a+=acenter
        for an in range(-1,2,1):
            an=an*step
            an+=ancenter
            for m in range(-1,2,1): 
                m=m*step
                m+=mcenter
                for r in range(-1,2,1): 
                    r=r*step
                    r+=rcenter
                    for mn in range(-1,2,1): 
                        mn=mn*step
                        mn+=mncenter

                        x0=(a,an,m,r,mn)

                        est=calculaEst(x0)
                        err=errorEstMed("estadisticosDiarios.csv",7,est,to_datetime("2019-05-15"),to_datetime("2019-06-20"))
                        errsq=err*err
                        sumerr=errsq.sum()
                        if sumerr<minerr:
                            bestfit=x0
                            minerr=sumerr
    if x1==bestfit:
        print("updating best fit")
        print(step)
        step=step/2
        print(x1)
        print(minerr)
    else:
        print("new best fit")
        print(step)
        print(bestfit)
        print(minerr)
        acenter=bestfit[0]
        ancenter=bestfit[1]
        mcenter=bestfit[2]
        rcenter=bestfit[3]
        mncenter=bestfit[4]
        x1=bestfit
x0=bestfit
est=calculaEst(x0)
print(est)
est=est.add_prefix("Estado estimado ")
estim=calculaEstim(x0)
estimMedio=estMed(estim)
estMedio=estimMedio.applymap(calculaEstado,min=2,max=3)

estim=estim.add_prefix("Estimador de estado ")
estimMedio=estimMedio.add_prefix("Estimador de estado medio ")
estMedio=estMedio.add_prefix("Estado medio ")
print(estimMedio)
ret=errorEstMed("estadisticosDiarios.csv",7,est,to_datetime("2019-05-15"),to_datetime("2019-06-20"))
print(ret)


# ret=error("estadisticosDiarios.csv",7,12,est,1)
# print(ret)
df=pd.read_csv("estadisticosDiarios.csv")
df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
df=df.drop_duplicates(subset="Fecha")
df.dropna(subset = ["Fecha"], inplace=True)
df=df.set_index("Fecha")
df=pd.merge(df,est,how="outer",on="Fecha")
df=pd.merge(df,estim,how="outer",on="Fecha")
df=pd.merge(df,estMedio,how="outer",on="Fecha")
df=pd.merge(df,estimMedio,how="outer",on="Fecha")
df.to_csv("estadisticosDiariosConEst2.csv", index=True)


#bestval=op.minimize(calculaEstSum,x0,method='nelder-mead',options={'return_all':True,'disp':True})
#print(bestval)



