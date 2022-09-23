from numpy import where
import pandas as pd
from pandas.core.tools.datetimes import to_datetime
import copy as cp

def calculaEstado(x,min,max):
    estado=2
    if x<min:
        estado=1
    if x>max:
        estado=3
    return estado

def error(df,primsen,est,sens):

    
    length=int(len(est.columns)/2)
    dfs=df.iloc[:,primsen:primsen+length]

    dfe1=est.iloc[:,sens:length*2:2]
    dfe1.dropna(inplace=True)
    dfs.dropna(inplace=True)
    dfs1=dfs
    dfs1.columns=dfe1.columns
    ret=abs(dfs1-dfe1).mean()

    return ret

def errorEstMed(df,primsen,est,fini,fend):

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
    # dfe1.dropna(inplace=True)
    # dfe2.dropna(inplace=True)
    dfe2.columns=dfe1.columns
    dfe=(dfe1+dfe2)/2

    return dfe

def calculaEstSum(dat,design_variables,pre,minexp,dia):#a,an,m,r,mn

    est=calculaEst(dat,design_variables,pre,minexp)

    ret=error(dia,7,12,est,0)
    ret=ret*ret
    ret=ret.sum()
    return ret

def calculaEstim(dat,design_variables,pre,minexp):#,fini,fend):#a,an,m,pre,mn,mx=10
    lpti=dat[0].astype(float)
    lptmax=dat[1].astype(float)
    eTo=dat[2].astype(float)

    deltaEstInd = pd.DataFrame(0,index=lpti.index,columns=lpti.columns)
    for x in range(len(design_variables)):
        for y in range(len(design_variables[x])):
            for z in range(len(design_variables[x][y])):
                deltaEstInd+=design_variables[x][y][z]*lpti.pow(x+minexp)*lptmax.pow(y+minexp)*eTo.pow(z+minexp)
    estInd=deltaEstInd
    for i in range(1, len(estInd)):
        estInd.iloc[i] = estInd.iloc[i-1] * pre + deltaEstInd.iloc[i]*(1-pre)
    return estInd

def calculaEst(dat,design_variables,pre,minexp):
    estInd=calculaEstim(dat,design_variables,pre,minexp)
    est=estInd.applymap(calculaEstado,min=2,max=3)
    return est

def cargaDatos():
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
    eTo=lpti
    for i in range(eTo.shape[1]):
        eTo.iloc[:,i]=dfd.loc[:,"ETo"]

    df=pd.read_csv("estadisticosDiarios.csv")
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    dia=df.set_index("Fecha")

    ret=[lpti,lptmax,eTo,dia]
    return ret

def printfun(bestfit,minexp,pre):
    st="("
    for x in range(len(bestfit)):
        for y in range(len(bestfit)):
            for z in range(len(bestfit)):
                if bestfit[x][y][z]!=0:
                    if bestfit[x][y][z]>0:
                        st+="+"
                    st+=str(bestfit[x][y][z])
                    if x+minexp!=0:
                        st+="a^("+str(x+minexp)+")"
                    if y+minexp!=0:
                        st+="max^("+str(y+minexp)+")"
                    if z+minexp!=0:
                        st+="ETo^("+str(z+minexp)+")"
    st+=")*"+str(1-pre)+"+fo*"+str(pre)
    return st

dat=cargaDatos()
lpti=dat[0]
lptmax=dat[1]
eTo=dat[2]
dia=dat[3]

x0=[]
for x in range(5):
    x0.append([])
    for y in range(5):
        x0[x].append([])
        for z in range(5):
            x0[x][y].append(0)

# x0 = np.zeros([5, 5, 5])
# x0[0][4][4]=-0.75
# x0[1][2][4]=-1
# x0[2][0][4]=-1
# x0[2][2][2]=1.25
# x0[3][1][2]=1.7
# x0[3][2][2]=-0.65

pre=-0.45
minexp=-2
#print(calculaEst(dat,x0,0.3,-1))

pre1=pre
pre0=pre
bestfit=cp.deepcopy(x0) #Mueve el centro al bestfit en cada iteración. si ya se encuentra alli, reduce el paso a la mitad.

x1=cp.deepcopy(x0)
step=1
minstep=0.01
minerr=100

while step>minstep:
    for a in range(-1,2,1): 
        x0=cp.deepcopy(x1)
        pre0=pre1+a*step
        est=calculaEst(dat,x0,pre0,minexp)
        err=error(dia,7,est,0)
        errsq=err*err
        sumerr=errsq.sum()
        # #print(str(x0)+": "+str(sumerr))
        # print(pre1)
        # print(a)
        # print(sumerr)
        # print(printfun(x0,minexp,pre0))
        # print(printfun(bestfit,minexp,pre))
        if sumerr<minerr:
            pre=pre0
            minerr=sumerr
    for x in range(len(x0)):
        for y in range(len(x0)):
            for z in range(len(x0)):
                for a in range(-1,2,2): 
                    x0=cp.deepcopy(x1)
                    x0[x][y][z]+=a*step
                    est=calculaEst(dat,x0,pre,minexp)
                    err=error(dia,7,est,0)
                    errsq=err*err
                    sumerr=errsq.sum()
                    #print(str(x0)+": "+str(sumerr))
                    if sumerr<minerr:
                        #print(sumerr)
                        bestfit=cp.deepcopy(x0)
                        minerr=sumerr
    if x1==bestfit and pre1==pre:
        print("updating step:"+str(step))
        step=step/2
    else:
        st=printfun(bestfit,minexp,pre)
        print("new best fit\nstep:"+str(step)+"\nfunction:"+str(st)+"\nerror:"+str(minerr))
        x1=cp.deepcopy(bestfit)
        pre1=pre
        
x0=cp.deepcopy(bestfit)
est=calculaEst(dat,x0,pre,minexp)
print(est)
est=est.add_prefix("Estado estimado ")
estim=calculaEstim(dat,x0,pre,minexp)
estimMedio=estMed(estim)
estMedio=estimMedio.applymap(calculaEstado,min=2,max=3)

estim=estim.add_prefix("Estimador de estado ")
estimMedio=estimMedio.add_prefix("Estimador de estado medio ")
estMedio=estMedio.add_prefix("Estado medio ")
print(estimMedio)
ret=error(dia,7,est,1)
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
df.to_csv("estadisticosDiariosConEst3.csv", index=True)


#bestval=op.minimize(calculaEstSum,x0,method='nelder-mead',options={'return_all':True,'disp':True})
#print(bestval)