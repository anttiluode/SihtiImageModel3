"""R4 preregistered directional flux ablation, transparent oracle and visual audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from data import train_set, held_out, damage, load_images
from flux import FluxMedium, image_features
from sihti3 import gaussian_fill, LearnedMedium

ARMS=("smooth","scalar","gradient","both","shuffled","oracle_gradient","oracle_scalar","oracle_both")


def evaluate(image,target,mask):
    hidden=~mask
    e=image-target
    dx=np.diff(e,axis=1); dy=np.diff(e,axis=0)
    xmask=hidden[:,1:]|hidden[:,:-1]
    ymask=hidden[1:,:]|hidden[:-1,:]
    edge=float((np.sum(dx[xmask]**2)+np.sum(dy[ymask]**2))/
               max(1,xmask.sum()+ymask.sum()))
    tx=np.diff(target,axis=1,append=target[:,-1:])
    ty=np.diff(target,axis=0,append=target[-1:,:])
    px=np.diff(image,axis=1,append=image[:,-1:])
    py=np.diff(image,axis=0,append=image[-1:,:])
    mag=np.hypot(tx,ty)
    strong=hidden&(mag>.08)
    cos=np.sum((tx*px+ty*py)[strong]/(np.maximum(mag[strong]*np.hypot(px[strong],py[strong]),1e-8))) / max(1,strong.sum())
    return {"mse":float(np.mean(e[hidden]**2)),
            "masked_gradient_mse":edge,"strong_edge_cosine":float(cos),
            "strong_edge_pixels":int(strong.sum())}


def messages(model,observation,mask,target,seed):
    ordinary,coverage=model.predict(observation,mask)
    truth=image_features(target)
    full=mask.reshape(8,4,8,4).mean((1,3)).ravel()
    affected=np.flatnonzero(full<1)
    rng=np.random.default_rng(seed)
    shuffle=ordinary.copy()
    permutation=rng.permutation(affected)
    shuffle[64+affected]=ordinary[64+permutation]
    shuffle[128+affected]=ordinary[128+permutation]
    ograd=ordinary.copy()
    ograd[64+affected]=truth[64+affected]
    ograd[128+affected]=truth[128+affected]
    osc=ordinary.copy()
    osc[affected]=truth[affected]
    oboth=ograd.copy()
    oboth[affected]=truth[affected]
    return ordinary,shuffle,ograd,osc,oboth,coverage,truth,full


def run(output="runs/flux_r4",train_folder=None,eval_folder=None,
        train_limit=320,eval_limit=32,seed=383,show_index=0,with_old=True):
    if (train_folder is None)!=(eval_folder is None):
        raise ValueError("Need both --train-folder and --eval-folder")
    if train_folder:
        train,_=load_images(train_folder,max_images=train_limit)
        unseen,_=load_images(eval_folder,max_images=eval_limit or None)
        oldhash={hashlib.sha256(x.tobytes()).digest() for x in train}
        if any(hashlib.sha256(x.tobytes()).digest() in oldhash for x in unseen):
            raise ValueError("Duplicate processed train/test image")
        domain="user supplied 32x32 grayscale photos"
    else:
        train=train_set(train_limit)
        unseen=held_out(eval_limit or 32,seed=seed)
        domain="disjoint synthetic contours"
    if not 0<=show_index<len(unseen):
        raise ValueError("Invalid show_index")
    model=FluxMedium.fit(train)
    old=LearnedMedium.fit(train,seed=2026) if with_old else None
    records=[];shown=None
    for i,target in enumerate(unseen):
        kind="gap" if i%2==0 else "hole"
        obs,mask=damage(target,seed=seed*10+i,kind=kind)
        m,sh,og,os,ob,coverage,truth,full=messages(model,obs,mask,target,seed+i)
        predictions={
            "smooth":model.solve(obs,mask,m,"smooth"),
            "scalar":model.solve(obs,mask,m,"scalar"),
            "gradient":model.solve(obs,mask,m,"gradient"),
            "both":model.solve(obs,mask,m,"both"),
            "shuffled":model.solve(obs,mask,sh,"both"),
            "oracle_gradient":model.solve(obs,mask,og,"both"),
            "oracle_scalar":model.solve(obs,mask,os,"both"),
            "oracle_both":model.solve(obs,mask,ob,"both"),
            "linear":gaussian_fill(obs,mask),
        }
        if old is not None:
            predictions["original_sihti3"]=old.run(obs,mask,steps=16)["image"]
        masks={
            "fully_hidden":full==0,
            "partially_visible":(full>0)&(full<1),
            "affected":full<1,
        }
        feature_errors={}
        for label,selector in masks.items():
            # Flux oracle quality; average only on mask-affected block cells.
            count=int(selector.sum())
            feature_errors[label]={
                "blocks":count,
                "gx_mse":float(np.mean((m[64:128][selector]-truth[64:128][selector])**2)) if count else None,
                "gy_mse":float(np.mean((m[128:][selector]-truth[128:][selector])**2)) if count else None,
                "dc_mse":float(np.mean((m[:64][selector]-truth[:64][selector])**2)) if count else None,
            }
        row={"index":i,"mask":kind,
             "scores":{key:evaluate(pred,target,mask) for key,pred in predictions.items()},
             "features":feature_errors}
        records.append(row)
        if i==show_index:
            shown=(target,obs,mask,predictions,m,truth)
    means={}
    for key in records[0]["scores"]:
        means[key]={metric:float(np.mean([x["scores"][key][metric] for x in records]))
                    for metric in ("mse","masked_gradient_mse","strong_edge_cosine")}
    improvement=lambda key,vs,metric:float(1-means[key][metric]/max(means[vs][metric],1e-12))
    gates={
        "both_beats_scalar_5pct_mse":improvement("both","scalar","mse")>=.05,
        "both_beats_scalar_5pct_gradient_error":improvement("both","scalar","masked_gradient_mse")>=.05,
        "both_beats_shuffled_5pct_mse":improvement("both","shuffled","mse")>=.05,
    }
    gates["all_pass"]=all(gates.values())
    report={"gate":"R4 directional 8x8 block flux, continuous decoder; no ping",
            "domain":domain,"n_train":len(train),"n_test":len(unseen),
            "results":means,"gains":{
                "both_vs_scalar_mse":improvement("both","scalar","mse"),
                "both_vs_scalar_masked_gradient":improvement("both","scalar","masked_gradient_mse"),
                "both_vs_shuffled_mse":improvement("both","shuffled","mse")
            },"gates":gates,"cases":records}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target,obs,mask,preds,m,truth=shown
    keys=["linear","scalar","gradient","both","shuffled","oracle_gradient","oracle_scalar","oracle_both"]
    panels=[("True (evaluation only)",target),("Observed / gap",obs)] + [
        (key.replace("_"," "),preds[key]) for key in keys]
    fig,axs=plt.subplots(2,5,figsize=(14,6))
    for ax,(label,img) in zip(axs.flat,panels):
        ax.imshow(img,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        ax.set_title(label,fontsize=9)
        ax.axis("off")
    fig.suptitle("R4 vector flux vs scalar brightness, with privileged oracles clearly labeled")
    fig.tight_layout()
    fig.savefig(out/"comparison.png",dpi=140)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    for ax,vec,label in zip(axs,(m,truth),("Conditioned vector flux","TRUE vector flux (evaluation only)")):
        gx=vec[64:128].reshape(8,8);gy=vec[128:].reshape(8,8)
        ax.quiver(np.arange(8)+.5,np.arange(8)+.5,gx,gy,scale=2)
        ax.set(xlim=(0,8),ylim=(8,0),title=label)
        ax.set_aspect("equal")
    fig.tight_layout();fig.savefig(out/"flux_vectors.png",dpi=140);plt.close(fig)
    start=preds["linear"];end=preds["both"];frames=[]
    for t in range(13):
        # Illustrative interpolation; NOT simulated gradient descent steps.
        a=(1-np.exp(-t/3))/(1-np.exp(-12/3))
        frame=(1-a)*start+a*end
        frame[mask]=obs[mask]
        pic=Image.fromarray(np.uint8(np.clip(frame,0,1)*255+.5))
        pic=pic.resize((320,320),Image.Resampling.NEAREST).convert("RGB")
        ImageDraw.Draw(pic).text((4,4),f"blend {t}/12 (not solver dynamics)",fill=(255,0,0))
        frames.append(pic)
    frames[0].save(out/"loop.gif",save_all=True,append_images=frames[1:],duration=250,loop=0)
    print(json.dumps({"results":means,"gains":report["gains"],"gates":gates,
                      "n_test":len(unseen),"output":str(out)},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="runs/flux_r4")
    p.add_argument("--train-folder")
    p.add_argument("--eval-folder")
    p.add_argument("--train-limit",type=int,default=320)
    p.add_argument("--eval-limit",type=int,default=32,help="0 = all unseen")
    p.add_argument("--seed",type=int,default=383)
    p.add_argument("--show-index",type=int,default=0)
    p.add_argument("--no-old",action="store_true",help="Skip original Sihti3 comparator")
    a=p.parse_args()
    run(a.output,a.train_folder,a.eval_folder,a.train_limit,a.eval_limit,a.seed,
        a.show_index,not a.no_old)
