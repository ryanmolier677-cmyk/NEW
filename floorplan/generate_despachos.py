import cv2, numpy as np
rng=np.random.default_rng(7)
U="/root/.claude/uploads/01553a60-b9bb-5987-831c-34fef29ff3a6/"
orig=cv2.imread(U+"e64cd8d6-image.jpg").astype(np.float32)
base=cv2.imread("/home/user/NEW/floorplan/oficina-molduras.jpg").astype(np.float32)
Hh,Ww=base.shape[:2]
f,cx,cy,h=834.,1008.,770.,1.5
C=np.array([0.56,4.81]); E={"back":np.array([-0.8,-0.6]),"right":np.array([0.6,-0.8])}
def wp(w,u,v):
    X,Z=C+u*E[w]; return np.array([cx+f*X/Z, cy+f*(h-v)/Z])
def fp(p,q):
    X,Z=C+p*E["back"]+q*E["right"]; return [cx+f*X/Z, cy+f*h/Z]
def bgr(hexs): hexs=hexs.lstrip('#'); return np.array([int(hexs[4:6],16),int(hexs[2:4],16),int(hexs[0:2],16)],np.float32)
def poly(pts):
    m=np.zeros((Hh,Ww),np.float32); cv2.fillPoly(m,[np.round(np.array(pts)).astype(np.int32)],1.0); return m

# protect mask from ORIGINAL photo (cords, railing, lamps, extinguisher, sockets)
g0=cv2.cvtColor(orig.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
dev=np.abs(g0-cv2.medianBlur(g0.astype(np.uint8),31))
prot=(dev>16).astype(np.uint8)
lb=(slice(600,860),slice(940,1080)); prot[lb]=(dev[lb]>34).astype(np.uint8)
rail=poly([[1272,950],[1720,1030],[1760,1512],[1272,1512]])
prot=np.maximum(prot,(rail>0)&(dev>7)).astype(np.uint8)
# extinguisher & sign (reddish)
b,gch,r=[orig[...,i] for i in range(3)]
prot=np.maximum(prot,(((r-b)>60)&((r-gch)>60)).astype(np.uint8))
prot=cv2.dilate(prot,np.ones((3,3),np.uint8))
prot=cv2.GaussianBlur(prot.astype(np.float32),(5,5),0)

gray=cv2.cvtColor(base.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
def wall_mask(w):
    if w=="back": m=poly([wp(w,0,0.10),wp(w,3.06,0.10),wp(w,3.06,2.86),wp(w,0,2.86)])
    else:
        m=poly([wp(w,0,0.10),wp(w,2.58,0.10),wp(w,2.58,3.0),wp(w,0,3.0)])
    lum=g0>95
    return m*lum*(1-rail)
def paint(img,w,hexs,ref=232.0,vmin=None):
    m=wall_mask(w)*(1-prot)
    if w=="right":
        m=np.maximum(m,poly([[1098,470],[1178,452],[1178,1052],[1098,1042]])*(g0>95)*(1-prot)*(1-rail))
    if vmin is not None: m=m*poly([wp(w,0,vmin)+np.array([-90,0]),wp(w,2.58,vmin),wp(w,2.58,3.2),wp(w,0,3.2)+np.array([-90,0])])
    m=cv2.GaussianBlur(m,(3,3),0)
    col=bgr(hexs)
    new=col[None,None,:]*(gray/ref)[...,None]
    return img*(1-m[...,None])+new*m[...,None]

# ---------- procedural textures ----------
def lowfreq(hh,ww,scale,seed):
    r=np.random.default_rng(seed).random((max(2,hh//scale),max(2,ww//scale))).astype(np.float32)
    r=cv2.resize(r,(ww,hh),interpolation=cv2.INTER_CUBIC); return (r-r.min())/(r.max()-r.min()+1e-6)
def make_rug(kw,kh,pal,seed,fringe=False,cell=6):
    R=np.random.default_rng(seed)
    field,border,a1,a2,ivory,dark,wash=[bgr(c) for c in pal]
    y,x=np.mgrid[0:kh,0:kw]
    d=np.minimum(np.minimum(x,y),np.minimum(kw-1-x,kh-1-y))
    K=np.zeros((kh,kw,3),np.float32); K[:]=field
    B=max(5,int(0.13*kw))+3
    K[d<1]=dark; K[(d>=1)&(d<2)]=a1; K[(d>=2)&(d<3)]=dark
    band=(d>=3)&(d<B); K[band]=border
    t=np.where(np.minimum(y,kh-1-y)==d,x,y).astype(np.float32); sacr=(d-3).astype(np.float32)
    P=max(6,(B-3)*2); bc=(B-3-1)/2
    mot=(np.abs((t%P)-P/2)+np.abs(sacr-bc))
    K[band&(mot<=bc)]=a1; K[band&(mot<=bc-1.5)]=a2; K[band&(mot<=0.8)]=ivory
    K[(d>=B)&(d<B+1)]=ivory; K[(d>=B+1)&(d<B+2)]=dark
    fld=d>=B+2
    fx0,fy0,fx1,fy1=B+2,B+2,kw-B-3,kh-B-3
    lat=(np.abs((x%7)-3)+np.abs((y%9)-4))
    K[fld&(lat==2)]=a1; K[fld&(lat==0)]=ivory
    cxm,cym=kw/2,kh/2; a=(fx1-fx0)*0.42; b=(fy1-fy0)*0.30
    m=np.abs(x-cxm)/a+np.abs(y-cym)/b
    rings=[dark,border,ivory,a2,a1,dark,ivory]
    for i,c in enumerate(rings):
        r=1.0-i*0.14
        K[fld&(m<r)]=c
    K[fld&(m<1.0)&(m>0.86)&((x+y)%4==0)]=ivory
    for (qx,qy) in [(fx0,fy0),(fx1,fy0),(fx0,fy1),(fx1,fy1)]:
        mc=np.abs(x-qx)/(a*0.55)+np.abs(y-qy)/(b*0.55)
        K[fld&(mc<1)]=border; K[fld&(mc<0.75)]=a2; K[fld&(mc<0.45)]=ivory
    n=lowfreq(kh,kw,max(4,kw//5),seed)
    tt=np.clip((n-0.25)/0.6,0,1)[...,None]*0.5
    K=K*(1-tt)+wash*tt
    ab=1+0.06*np.sin(np.cumsum(R.normal(0,0.25,kh)))[:,None,None]
    K=K*ab*(1+R.normal(0,0.035,(kh,kw,1)))
    gK=K.mean(axis=2,keepdims=True); K=K*0.72+gK*0.28
    K=K*0.85+wash*0.15
    img=cv2.resize(K,(kw*cell,kh*cell),interpolation=cv2.INTER_NEAREST)
    yy,xx=np.mgrid[0:cell,0:cell]; knot=(0.95+0.05*np.cos((xx-cell/2)/cell*np.pi)*np.cos((yy-cell/2)/cell*np.pi)).astype(np.float32)
    img*=np.tile(knot,(kh,kw))[...,None]
    img=cv2.GaussianBlur(img,(0,0),cell*0.3)
    img+=R.normal(0,5,img.shape[:2]+(1,)).astype(np.float32)
    if fringe:
        fh=int(0.03*img.shape[0]); fr=np.zeros((fh,img.shape[1],3),np.float32); fr[:]=bgr("#E3D8C3")
        for xx2 in range(0,img.shape[1],5): fr[:,xx2:xx2+2]*=0.82
        img=np.concatenate([fr,img,fr],0)
    return np.clip(img,0,255)
def make_art(w,h,seed):
    R=np.random.default_rng(seed)
    img=np.zeros((h,w,3),np.float32); img[:]=bgr("#D9CCB4")
    def oct(sd):
        return sum(lowfreq(h,w,max(2,w//k),sd+k)*wt for k,wt in ((3,0.5),(7,0.3),(18,0.2)))
    for col,th,al in [("#C2A27A",0.50,0.7),("#E9E1D2",0.58,0.8),("#A8825E",0.66,0.6),("#6B5646",0.74,0.5),("#F1ECE2",0.70,0.6)]:
        n=oct(int(R.integers(1e6))); mm=np.clip((n-th)/0.10,0,1)[...,None]*al
        img=img*(1-mm)+bgr(col)*mm
    streak=cv2.blur(R.normal(0,1,(h,w)).astype(np.float32),(1,int(h*0.12)))
    img+=streak[...,None]*10+R.normal(0,4,(h,w,1)).astype(np.float32)
    img=cv2.GaussianBlur(img,(0,0),1.2)
    fr=max(5,w//45); img[:fr]=img[-fr:]=bgr("#2B2420"); img[:,:fr]=img[:,-fr:]=bgr("#2B2420")
    return np.clip(img,0,255)

def place(img,tex,quad,mask_extra=None,shade="wall",shadow=True,use_prot=True):
    th,tw=tex.shape[:2]
    src=np.float32([[0,0],[tw,0],[tw,th],[0,th]]); dst=np.float32(quad)
    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(tex,M,(Ww,Hh),flags=cv2.INTER_LINEAR)
    m=cv2.warpPerspective(np.ones((th,tw),np.float32),M,(Ww,Hh))
    if shade=="wall":
        L=np.clip(cv2.GaussianBlur(gray,(0,0),9)/232.0,0.55,1.15)
    else:
        L=np.clip(cv2.GaussianBlur(gray,(0,0),15)/np.median(gray[m>0.5]),0.6,1.25)
    if shadow:
        sh=cv2.GaussianBlur(np.roll(np.roll(m,6,0),4,1),(0,0),6)*(1-m)
        img=img*(1-0.28*sh[...,None])
    keep=(1-prot) if use_prot else np.ones_like(prot)
    if mask_extra is not None: keep=keep*mask_extra
    mm=(m*keep)[...,None]
    return img*(1-mm)+warped*L[...,None]*mm

def lay(a,b,n,gap):
    w=(b-a-gap*(n-1))/n; return [(a+i*(w+gap),a+i*(w+gap)+w) for i in range(n)]
BACK=lay(0.29,2.806,3,0.18); RIGHT=lay(0.63,2.55,3,0.18)
def wquad(w,u0,u1,v0,v1):
    # texture top-left -> image; for back wall u increases to the left
    A,B=(u1,u0) if w=="back" else (u0,u1)
    return [wp(w,A,v1),wp(w,B,v1),wp(w,B,v0),wp(w,A,v0)]
def fquad(p0,p1,q0,q1):
    # far edge (q0) at top of texture
    return [fp(p1,q0),fp(p0,q0),fp(p0,q1),fp(p1,q1)]

RUGS={
 "rust":["#8E4E36","#36415A","#C79A62","#5D6B5A","#E4D6BC","#2A231F","#B99A7E"],
 "indigo":["#3F4A62","#8E5A3E","#C9A57A","#7D8A80","#E2D7C2","#222226","#8C93A0"],
 "sand":["#C8B497","#7A4E36","#4F5A6E","#9C6B4C","#EDE3D1","#3D322B","#D9CDB8"],
 "sage":["#7F8468","#6B4535","#CDB58E","#3F4A5A","#E2D9C4","#2F2C26","#A9A891"],
 "rose":["#A46F5E","#3F4658","#DCC5A5","#6C7A69","#E8DAC6","#33292A","#C3A595"],
}
def floor_rug(img,pal,seed):
    tex=make_rug(130,150,RUGS[pal],seed,fringe=True,cell=6)
    return place(img,tex,fquad(0.85,2.85,0.2,2.6),shade="floor",shadow=False,use_prot=False)

def panel_rugs(img,pals):
    for (u0,u1),pal,s in zip(RIGHT,pals,(11,12,13)):
        tex=make_rug(46,170,RUGS[pal],s,cell=6)
        img=place(img,tex,wquad("right",u0+0.07,u1-0.07,0.95,2.43))
    return img
def art(img,seed=5):
    u0,u1=BACK[1]; tex=make_art(520,900,seed)
    return place(img,tex,wquad("back",u0+0.10,u1-0.10,1.05,2.33))
def big_rugs(img):
    a=make_rug(80,120,RUGS["sand"],21); b=make_rug(80,120,RUGS["rust"],22)
    img=place(img,a,wquad("right",0.55,1.55,1.05,2.6))
    img=place(img,b,wquad("right",1.35,2.4,0.95,2.5))
    # brass rods
    for (u0,u1,v) in [(0.5,1.6,2.62),(1.3,2.45,2.52)]:
        p0,p1=wp("right",u0,v),wp("right",u1,v)
        cv2.line(img,tuple(map(int,p0)),tuple(map(int,p1)),(70,140,185),5,cv2.LINE_AA)
    return img

out={}
# Option 1: warm white everywhere, rugs framed in panels, art on RE wall
im=paint(base.copy(),"back","#EEE6D9"); im=paint(im,"right","#EEE6D9")
im=floor_rug(im,"rust",31); im=panel_rugs(im,["indigo","rust","sand"]); im=art(im)
out[1]=im
# Option 2: RE wall deep taupe, rug wall warm white
im=paint(base.copy(),"back","#8E7D6C"); im=paint(im,"right","#EDE5D8")
im=floor_rug(im,"rose",32); im=panel_rugs(im,["sage","rose","rust"]); im=art(im,9)
out[2]=im
# Option 3: rug wall dark tobacco, RE wall cream; big layered rugs on brass rods
im=paint(base.copy(),"back","#ECE3D3"); im=paint(im,"right","#4B3A2F",vmin=0.86)
im=floor_rug(im,"sand",33); im=big_rugs(im); im=art(im,13)
out[3]=im
for k,v in out.items():
    v=np.clip(v,0,255).astype(np.uint8)
    cv2.imwrite(f"/home/user/NEW/floorplan/despachos-opcion-{k}.jpg",v,[cv2.IMWRITE_JPEG_QUALITY,92])
    cv2.imwrite(f"op{k}_small.jpg",cv2.resize(v,(1100,825),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,82])
print("ok")
