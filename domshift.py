import config  # ortak yollar ve SMOKE modu
import numpy as np, pandas as pd, time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
REG=["eps_inf","delta_eps1","tau1_ps","alpha1","sigma"]
def load(fp):
    df=pd.read_csv(fp); idc=["tissue"]+REG+["snr_db"]
    df=df.sort_values(idc+["f_GHz"]); X=[];y=[]
    for k,g in df.groupby(idc,sort=False):
        s=g.sort_values("f_GHz")["S21_Abs"].values
        if len(s)==111: X.append(s); y.append(k[0])
    return np.array(X,dtype=np.float32), np.array(y)
t=time.time()
X,y=load(config.ONEPOLE_CSV); Xv,yv=load(config.VAL4_CSV)
le=LabelEncoder().fit(y); ye=le.transform(y); yve=le.transform(yv); print("yüklendi %.0fs"%(time.time()-t), X.shape, Xv.shape)
idx=np.arange(len(ye)); tr,te=train_test_split(idx,test_size=.2,random_state=42,stratify=ye)
fit,va=train_test_split(tr,test_size=.1,random_state=42,stratify=ye[tr])
sc=StandardScaler().fit(X[fit]); svm=SVC(kernel="rbf",C=10,gamma="scale").fit(sc.transform(X[fit]),ye[fit])
pt=svm.predict(sc.transform(X[te])); pv=svm.predict(sc.transform(Xv))
print("\nSVM (ham S21, TAM eğitim seti, seed42)  hold-out test acc = %.4f | 4-kutuplu doğrulama acc = %.4f"%(accuracy_score(ye[te],pt),accuracy_score(yve,pv)))
print("\nDoku başına 4-kutuplu doğruluk ve en sık tahmin edilen sınıf:")
for c in le.classes_:
    m=yv==c; k=le.transform([c])[0]; a=(pv[m]==k).mean()
    top=pd.Series(le.inverse_transform(pv[m])).value_counts().head(2)
    print(f"  {c:20s} acc={a:5.2f}  -> "+", ".join(f"{i}:{n/m.sum():.2f}" for i,n in top.items()))
