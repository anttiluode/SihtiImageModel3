"""R4 signed vector flux and continuous quadratic image inverse (not patch MRF)."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sihti3 import gaussian_fill, sieve

N, G, B = 32, 8, 4
F = 3*G*G


def block(x):
    return np.asarray(x).reshape(G, B, G, B)


def image_features(image):
    p = block(image)
    return np.concatenate((p.mean((1,3)).ravel(),
                           np.diff(p, axis=3).mean((1,3)).ravel(),
                           np.diff(p, axis=1).mean((1,3)).ravel()))


def observed_features(observation, mask, fallback=None):
    """Gradients require TWO visible endpoints, no hidden value leakage."""
    mask = np.asarray(mask, bool)
    if mask.shape != (N,N) or np.asarray(observation).shape != (N,N):
        raise ValueError("expected 32x32")
    p = block(np.where(mask, observation, 0.))
    m = block(mask)
    def estimate(x, valid, total):
        counts = valid.sum(axis=(1,3))
        y = np.sum(np.where(valid, x, 0.), axis=(1,3))
        vals = np.divide(y, counts, out=np.zeros_like(y, dtype=float),
                         where=counts>0).ravel()
        return vals, (counts/total).ravel()
    dc, dc_n = estimate(p, m, 16)
    dx, dx_n = estimate(np.diff(p,axis=3),
                       m[:,:,:,1:] & m[:,:,:,:-1], 12)
    dy, dy_n = estimate(np.diff(p,axis=1),
                       m[:,1:,:,:] & m[:,:-1,:,:], 12)
    values = np.concatenate((dc,dx,dy))
    n = np.concatenate((dc_n,dx_n,dy_n))
    if fallback is not None:
        values = np.where(n>0, values, fallback)
    return values, n


def operators():
    rr=[]; cc=[]; vv=[]
    def push(r,y,x,v):
        rr.append(r); cc.append(y*N+x); vv.append(v)
    for by in range(G):
        for bx in range(G):
            ix=by*G+bx
            for y in range(by*B,(by+1)*B):
                for x in range(bx*B,(bx+1)*B):
                    push(ix,y,x,1/16)
                    if x%4<3:
                        push(64+ix,y,x,-1/12);push(64+ix,y,x+1,1/12)
                    if y%4<3:
                        push(128+ix,y,x,-1/12);push(128+ix,y+1,x,1/12)
    f=sparse.coo_matrix((vv,(rr,cc)),shape=(F,N*N)).tocsr()
    rr=[];cc=[];vv=[];row=0
    for y in range(N):
        for x in range(N):
            if x+1<N:
                rr.extend([row,row]);cc.extend([y*N+x,y*N+x+1]);vv.extend([-1,1]);row+=1
            if y+1<N:
                rr.extend([row,row]);cc.extend([y*N+x,(y+1)*N+x]);vv.extend([-1,1]);row+=1
    d=sparse.coo_matrix((vv,(rr,cc)),shape=(row,N*N)).tocsr()
    return f,d


F_OPERATOR,D_OPERATOR=operators()


@dataclass
class FluxMedium:
    mean: np.ndarray
    scale: np.ndarray
    precision: np.ndarray
    observation_strength: float=10.0
    smooth_weight: float=.10
    scalar_weight: float=5.0
    gradient_weight: float=55.0
    anchor_weight: float=.002

    @classmethod
    def fit(cls, train):
        train=np.asarray(train,float)
        if train.ndim!=3 or train.shape[1:]!=(32,32) or len(train)<24:
            raise ValueError("Need >=24 training 32x32 grayscale images")
        x=np.asarray([image_features(im) for im in train])
        mu=x.mean(0)
        std=np.maximum(x.std(0),.035)
        z=(x-mu)/std
        cov=.72*np.cov(z,rowvar=False)+.28*np.eye(F)
        return cls(mu,std,np.linalg.inv(cov))

    def predict(self, observation, mask):
        measured,coverage=observed_features(observation,mask,self.mean)
        norm=(measured-self.mean)/self.scale
        w=self.observation_strength*coverage**2
        posterior=np.linalg.solve(self.precision+np.diag(w),w*norm)
        return self.mean+self.scale*posterior,coverage

    def solve(self, observation, mask, message, mode="both"):
        """Exact free-pixel solution of a continuous convex gradient energy.

        E(u)=.5*a*||D u||²+.5*sum_k w_k(F_k u - message_k)²
             +.5*epsilon*||u - interpolation||².
        """
        if mode not in ("smooth","scalar","gradient","both"):
            raise ValueError(mode)
        observed=np.asarray(mask,bool)
        data=np.asarray(observation,float)
        msg=np.asarray(message,float).ravel()
        if observed.shape!=(N,N) or data.shape!=(N,N) or msg.shape!=(F,) or not observed.any() or observed.all():
            raise ValueError("32x32 partially masked input and 192-feature message required")
        start=gaussian_fill(np.where(observed,data,0),observed)
        weight=np.zeros(F)
        if mode in ("scalar","both"): weight[:64]=self.scalar_weight
        if mode in ("gradient","both"): weight[64:]=self.gradient_weight
        a=(self.smooth_weight*(D_OPERATOR.T@D_OPERATOR)
           +F_OPERATOR.T@sparse.diags(weight)@F_OPERATOR
           +self.anchor_weight*sparse.eye(N*N)).tocsr()
        rhs=F_OPERATOR.T@(weight*msg)+self.anchor_weight*start.ravel()
        visible=np.flatnonzero(observed.ravel())
        hidden=np.flatnonzero(~observed.ravel())
        out=np.zeros(N*N)
        out[visible]=data.ravel()[visible]
        b=rhs[hidden] - a[hidden][:,visible]@out[visible]
        out[hidden]=spsolve(a[hidden][:,hidden].tocsc(),b)
        return np.clip(out.reshape(N,N),0.,1.)

    def response(self, observation, mask, message, kick=.1):
        """Separated differential echo of a gradient-channel perturbation.

        This diagnostic measures susceptibility, not an improvement to images.
        """
        baseline=self.solve(observation,mask,message,"both")
        shifted=np.asarray(message,float).copy()
        affected=np.flatnonzero((~np.asarray(mask,bool)).reshape(G,B,G,B).any((1,3)).ravel())
        if not len(affected):
            raise ValueError("No masked pixels to probe")
        cell=int(affected[len(affected)//2])
        shifted[64+cell]+=kick
        shifted[128+cell]-=kick
        return self.solve(observation,mask,shifted,"both")-baseline


def sieve_residues(image):
    return sieve(np.asarray(image,float))
