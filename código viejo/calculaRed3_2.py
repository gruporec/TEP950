import pandas as pd
import copy as cp

def calculaEstado(x,min,max):
    '''Devuelve un valor de estado (1 a 3) según estimador'''
    estado=2
    if x<min:
        estado=1
    if x>max:
        estado=3
    return estado

def error(data,est):
    '''Error absoluto cometido en cada instante.'''
    #recorta la estimación en función de los datos de comprobación
    idx = est.index.intersection(data.index)
    est1 = est.loc[idx]
    est1=est[data.columns]

    ret=abs(data-est1)

    return ret

# def errorEstMed(df,primsen,est,fini,fend): #DEPRECATED

#     df=df.loc[fini:fend]

#     length=int(len(est.columns)/2)

#     est.loc[fini:fend]
#     dfe1=est.iloc[:,0:length*2:2]
#     dfe2=est.iloc[:,1:length*2:2]
#     dfe1.dropna(inplace=True)
#     dfe2.dropna(inplace=True)
#     dfe2.columns=dfe1.columns
#     dfe=(dfe1+dfe2)/2
#     dfs=df.iloc[:,primsen:primsen+length]
#     dfs.dropna(inplace=True)
#     dfs.columns=dfe.columns
#     ret=abs(dfs-dfe).mean()

#     return ret

def estMed(est): #unused
    length=int(len(est.columns)/2)
    dfe1=est.iloc[:,0:length*2:2]
    dfe2=est.iloc[:,1:length*2:2]
    # dfe1.dropna(inplace=True)
    # dfe2.dropna(inplace=True)
    dfe2.columns=dfe1.columns
    dfe=(dfe1+dfe2)/2

    return dfe

def calculaEstSqErr(input,design_variables,pre,minexp,data): #unused

    est=calculaEst(input,design_variables,pre,minexp)

    ret=error(data,est,0)
    ret=ret*ret
    ret=ret.values.mean()
    return ret

def calculaEstim(dat,design_variables,pre,minexp):#,fini,fend):#a,an,m,pre,mn,mx=10
    '''Calcula un estimador del estado hidrico'''
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

    df=pd.read_csv("calibracion.csv")
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    cal=df.set_index("Fecha")

    df=pd.read_csv("validacion.csv")
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
    df=df.drop_duplicates(subset="Fecha")
    val=df.set_index("Fecha")

    ret=[lpti,lptmax,eTo,cal,val] #lpti, lptmax, eTo, datos calibracion, datos verificacion
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
cal=dat[3]
val=dat[4]
datval=[lpti,lptmax,eTo,val]

lptical=lpti.loc[cal.index,cal.columns]
lptmaxcal=lpti.loc[cal.index,cal.columns]
eTocal=lpti.loc[cal.index,cal.columns]
datcal=[lptical,lptmaxcal,eTocal,cal]

# x0=[]
# for x in range(5):
#     x0.append([])
#     for y in range(5):
#         x0[x].append([])
#         for z in range(5):
#             x0[x][y].append(0)
x0=[[[124609.375, -1073.125, -48.125, 11, 0], [-156.25, 0, 0, 0, 5], [50, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[-2812.5, 30, 0, 0, 0], [70, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]

# x0 = np.zeros([5, 5, 5])
# x0[0][4][4]=-0.75
# x0[1][2][4]=-1
# x0[2][0][4]=-1
# x0[2][2][2]=1.25
# x0[3][1][2]=1.7
# x0[3][2][2]=-0.65

pre=0
minexp=-2
#print(calculaEst(dat,x0,0.3,-1))

pre1=pre
pre0=pre
bestfit=cp.deepcopy(x0) #Mueve el centro al bestfit en cada iteración. si ya se encuentra alli, reduce el paso a la mitad.

x1=cp.deepcopy(x0)
step=78.125
minstep=0.01
minerr=100

while step>minstep:
    for x in range(len(x0)):
        for y in range(len(x0[x])):
            for z in range(len(x0[x][y])):
                for a in range(-1,2,2): 
                    x0=cp.deepcopy(x1)
                    x0[x][y][z]+=a*step
                    est=calculaEstim(datcal,x0,pre1,minexp)
                    err=error(cal,est)
                    errsq=err.pow(2).mean()
                    sumerr=errsq.mean()
                    if sumerr<minerr:
                        #print(sumerr)
                        bestfit=cp.deepcopy(x0)
                        minerr=sumerr
                    #print(str(x0)+": \n"+str(sumerr)+"\n"+str(minerr))
    for a in range(-1,2,1): 
        x0=cp.deepcopy(bestfit)
        pre0=pre1+a*step
        est=calculaEstim(datcal,x0,pre0,minexp)
        err=error(cal,est)
        errsq=err.pow(2).mean()
        sumerr=errsq.mean()
        if sumerr<minerr:
            pre=pre0
            minerr=sumerr
    if x1==bestfit and pre1==pre:
        print("updating step:"+str(step))
        step=step/2
    else:
        st=printfun(bestfit,minexp,pre)
        print("new best fit\nstep:"+str(step)+"\nfunction:"+str(st)+"\nerror:"+str(minerr)+"\nx0:"+str(x0))
        x1=cp.deepcopy(bestfit)
        pre1=pre
        
x0=cp.deepcopy(bestfit)
est=calculaEst(datval,x0,pre,minexp)
print(est)
est=est.add_prefix("Estado estimado ")
estim=calculaEstim(datval,x0,pre,minexp)
#estimMedio=estMed(estim)
#estMedio=estimMedio.applymap(calculaEstado,min=2,max=3)

estim=estim.add_prefix("Estimador de estado ")
#estimMedio=estimMedio.add_prefix("Estimador de estado medio ")
#estMedio=estMedio.add_prefix("Estado medio ")
#print(estimMedio)
# print(estim)
# ret=error(val,estim)
# print(ret.mean())


# ret=error("estadisticosDiarios.csv",7,12,est,1)
# print(ret)
# df=pd.read_csv("estadisticosDiarios.csv")
# df.iloc[:,0]=pd.to_datetime(df.iloc[:,0]) # Fecha como datetime
# df=df.drop_duplicates(subset="Fecha")
# df.dropna(subset = ["Fecha"], inplace=True)
# df=df.set_index("Fecha")
df=pd.merge(val,est,how="outer",on="Fecha")
df=pd.merge(df,estim,how="outer",on="Fecha")
#df=pd.merge(df,estMedio,how="outer",on="Fecha")
#df=pd.merge(df,estimMedio,how="outer",on="Fecha")
df.to_csv("resultadosEst.csv", index=True)

with open('x0.txt', 'w') as f:
    f.write(str(x0))

#bestval=op.minimize(calculaEstSum,x0,method='nelder-mead',options={'return_all':True,'disp':True})
#print(bestval)