import cv2, numpy as np
U="/root/.claude/uploads/01553a60-b9bb-5987-831c-34fef29ff3a6/"
org=cv2.imread(U+"e64cd8d6-image.jpg").astype(np.float32)
Hh,Ww=org.shape[:2]
f,cx,cy,h=834.,1008.,770.,1.5
C=np.array([0.56,4.81])
walls={"back":np.array([-0.8,-0.6]),"right":np.array([0.6,-0.8])}
def proj(wall,u,v):
    X,Z=C+u*walls[wall]; Y=h-v
    return np.array([cx+f*X/Z, cy+f*Y/Z])
def layout(a,b,n,gap):
    w=(b-a-gap*(n-1))/n
    return [(a+i*(w+gap),a+i*(w+gap)+w) for i in range(n)]
rows=[(0.22,0.70),(0.88,2.50)]
panels=[]
for wall,cols in (("back",layout(0.29,2.806,3,0.18)),("right",layout(0.63,2.55,3,0.18))):
    for (u0,u1) in cols:
        for (v0,v1) in rows: panels.append((wall,u0,u1,v0,v1))

# profile strips: (t0,t1, angle deg, facing) facing +1 = outward, -1 inward
prof=[(0,0.004,70,1),(0.004,0.012,35,1),(0.012,0.016,12,1),(0.016,0.028,0,0),
      (0.028,0.031,-50,1),(0.031,0.034,40,1),(0.034,0.045,32,-1)]
L=np.array([-0.25,0.6,0.8]); L/=np.linalg.norm(L)
def shade(o,ang,face):
    a=np.radians(abs(ang)); s=1 if ang>=0 else -1
    n=np.array([o[0]*np.sin(a)*face*s, o[1]*np.sin(a)*face*s, np.cos(a)])
    return (0.5+0.5*max(n@L,0))/(0.5+0.5*L[2])
SS=3
sh=np.ones((Hh*SS,Ww*SS),np.float32)
band=np.zeros((Hh*SS,Ww*SS),np.uint8)
shadow=np.zeros((Hh,Ww),np.float32)
for wall,u0,u1,v0,v1 in panels:
    # image-space orientation: which u direction is image-right
    ur=1 if proj(wall,u0+0.1,1)[0]>proj(wall,u0,1)[0] else -1
    def P(u,v): return proj(wall,u,v)*SS
    ul,urr=(u0,u1) if ur==1 else (u1,u0)   # image-left/right u
    sgn=ur
    for (t0,t1,ang,face) in prof:
        # edges: top, bottom, left, right with image outward dirs
        quads={
         "top":   ([P(ul-0*sgn+t0*sgn,v1-t0),P(urr-t0*sgn,v1-t0),P(urr-t1*sgn,v1-t1),P(ul+t1*sgn,v1-t1)],(0,1)),
         "bottom":([P(ul+t0*sgn,v0+t0),P(urr-t0*sgn,v0+t0),P(urr-t1*sgn,v0+t1),P(ul+t1*sgn,v0+t1)],(0,-1)),
         "left":  ([P(ul+t0*sgn,v0+t0),P(ul+t0*sgn,v1-t0),P(ul+t1*sgn,v1-t1),P(ul+t1*sgn,v0+t1)],(-1,0)),
         "right": ([P(urr-t0*sgn,v0+t0),P(urr-t0*sgn,v1-t0),P(urr-t1*sgn,v1-t1),P(urr-t1*sgn,v0+t1)],(1,0)),
        }
        for k,(q,o) in quads.items():
            val=shade(o,ang,face) if face!=0 else 1.025
            pts=np.round(np.array(q)).astype(np.int32)
            cv2.fillPoly(sh,[pts],float(val)); cv2.fillPoly(band,[pts],255)
    # cast shadow: outer rect shifted down-right ~1cm
    outer=np.array([proj(wall,u0,v0),proj(wall,u1,v0),proj(wall,u1,v1),proj(wall,u0,v1)])
    inner=np.array([proj(wall,u0+0.045,v0+0.045),proj(wall,u1-0.045,v0+0.045),proj(wall,u1-0.045,v1-0.045),proj(wall,u0+0.045,v1-0.045)])
    vs=abs(proj(wall,(u0+u1)/2,1.0)[1]-proj(wall,(u0+u1)/2,1.01)[1])  # px per cm
    m=np.zeros((Hh,Ww),np.float32)
    cv2.fillPoly(m,[np.round(outer).astype(np.int32)],1.0); cv2.fillPoly(m,[np.round(inner).astype(np.int32)],0.0)
    Mt=np.float32([[1,0,0.5*vs],[0,1,1.2*vs]])
    m=cv2.warpAffine(m,Mt,(Ww,Hh))
    k=int(max(3,vs*1.5))|1
    shadow=np.maximum(shadow,cv2.GaussianBlur(m,(k,k),0))

sh=cv2.resize(sh,(Ww,Hh),interpolation=cv2.INTER_AREA)
cov=cv2.resize(band,(Ww,Hh),interpolation=cv2.INTER_AREA).astype(np.float32)/255
shadow=shadow*(1-cov)
fac=sh*(1-shadow*0.16)
# keep foreground objects (railing, cords, lamps) untouched
g=cv2.cvtColor(org.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
dev=np.abs(g-cv2.medianBlur(g.astype(np.uint8),31).astype(np.float32))
prot=(dev>16).astype(np.uint8)
rail=np.zeros_like(prot); cv2.fillPoly(rail,[np.array([[1290,960],[1660,1030],[1660,1512],[1290,1512]],np.int32)],1)
prot=np.maximum(prot,rail*(dev>8))
prot=cv2.dilate(prot,np.ones((3,3),np.uint8))
prot=cv2.GaussianBlur(prot.astype(np.float32),(3,3),0)
fac=fac*(1-prot)+prot
# slightly whiten the molding (fresh paint) toward neutral white
out=org*fac[...,None]
whit=cov[...,None]*0.25
lum=out.mean(axis=2,keepdims=True)
out=out*(1-whit*(1-prot[...,None]))+lum*whit*(1-prot[...,None])
out=np.clip(out,0,255).astype(np.uint8)
cv2.imwrite("/home/user/NEW/floorplan/oficina-molduras.jpg",out,[cv2.IMWRITE_JPEG_QUALITY,93])
cv2.imwrite("zoom1.png",out[480:1150,480:1100]); cv2.imwrite("zoom2.png",out[380:1150,1100:1700])
print("done")
