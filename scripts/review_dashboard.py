"""Self-contained interactive review page for the study results.

Writes results/review/index.html (data embedded; figure gallery links to
../figures/*.png). Re-run after new results land:
    python -m scripts.review_dashboard
"""
import json
import os
import time

import numpy as np
import pandas as pd

from scripts.leaderboard import GEN_LABEL, compute, family, label, split_method

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
OUT = os.path.join(ROOT, "results", "review")

METRICS = {
    "tstr_catboost_auroc": ("TSTR AUROC (CatBoost)", "up"),
    "tstr_catboost_f1_macro": ("TSTR macro-F1 (CatBoost)", "up"),
    "alpha_precision": ("α-precision", "up"),
    "beta_recall": ("β-recall", "up"),
    "c2st_auc": ("C2ST detection AUC", "half"),
    "P": ("Privacy score P", "up"),
    "mia_domias_auc": ("DOMIAS MIA AUC", "half"),
    "mia_dist_auc": ("Distance MIA AUC", "half"),
    "aia_risk": ("Attribute-inference risk", "down"),
    "link_risk": ("Linkability risk", "down"),
    "share_closer_train": ("Share closer to train", "half"),
    "js_mean": ("Mean JS divergence", "down"),
    "repr_parity": ("Representation parity", "up"),
    "stat_parity_diff_max": ("Max. statistical-parity gap", "down"),
    "minority_class_ratio": ("Minority-class share ratio", "one"),
    "subgroup_util_ratio_min": ("Worst-subgroup utility ratio", "up"),
    "subgroup_util_ratio_mean": ("Mean subgroup utility ratio", "up"),
    "uniqueness": ("Uniqueness", "up"),
    "category_coverage": ("Category coverage", "up"),
    "rare_category_coverage": ("Rare-category coverage", "up"),
    "entropy_preservation": ("Entropy preservation", "up"),
    "tail_preservation": ("Tail preservation", "one"),
}
FAIR = ["repr_parity", "stat_parity_diff_max", "minority_class_ratio", "subgroup_util_ratio_min",
        "subgroup_util_ratio_mean", "rare_category_coverage", "category_coverage", "entropy_preservation",
        "tail_preservation", "uniqueness", "js_mean"]
ALL_GENS = ["rtf", "ctabgan_plus", "bayesian_network", "arf", "tvae", "ctgan", "tabddpm"]
DP = [f"{f}_eps{e}" for f in ("pategan", "ctabgan_plus_dp", "ctabgan_plus_dpgp") for e in (1, 3, 10)]
DS_LABEL = {"breast_cancer": "Breast Cancer", "diabetes": "Diabetes", "heart_failure": "Heart Failure",
            "hypothyroid": "Hypothyroid", "liver_disorders": "Liver Disorders", "obesity": "Obesity",
            "parkinsons": "Parkinson's", "adult": "Adult", "pir_vision": "PIR Vision", "vn_banking": "VN Banking"}


def build_payload():
    df = pd.read_csv(os.path.join(TAB, "all_runs_long.csv"))
    df = df[~df.method.str.endswith("+filter")]
    mets = [m for m in METRICS if m in df]
    g = df.groupby(["method", "dataset"])[mets]
    mean, sd, n = g.mean(), g.std(), g.size()
    cells = []
    for (m, d), row in mean.iterrows():
        rec = {"method": m, "dataset": d, "n": int(n.loc[(m, d)])}
        for k in mets:
            rec[k] = None if pd.isna(row[k]) else round(float(row[k]), 4)
            s = sd.loc[(m, d), k]
            rec[k + "_sd"] = None if pd.isna(s) else round(float(s), 4)
        cells.append(rec)
    lb, _ = compute(df)
    board = []
    for m, r in lb.iterrows():
        rec = {"method": m, "label": r.label, "family": r.family, "n_datasets": int(r.n_datasets)}
        for k in mets + ["rank_utility", "rank_privacy", "rank_fairness", "rank_overall"]:
            rec[k] = None if pd.isna(r.get(k, np.nan)) else round(float(r[k]), 4)
        board.append(rec)
    tr = pd.read_csv(os.path.join(TAB, "trtr_reference.csv"))
    trtr = {r.dataset: round(float(r.trtr_catboost_auroc), 4) for r in tr.itertuples()}
    trtr["__all__"] = round(float(tr.trtr_catboost_auroc.mean()), 4)
    present = set(df.method)
    methods = {m: {"label": label(m), "family": family(m), "gen": split_method(m)[0]} for m in present}
    pending = [GEN_LABEL[g] for g in ALL_GENS if g not in present]
    pending += [label(m) for m in DP if m not in present]
    figs = sorted(f for f in os.listdir(FIG) if f.endswith(".png")) if os.path.isdir(FIG) else []
    return {"cells": cells, "board": board, "trtr": trtr, "methods": methods, "pending": pending,
            "metrics": {k: {"label": v[0], "dir": v[1]} for k, v in METRICS.items() if k in mets},
            "fair": [k for k in FAIR if k in mets],
            "datasets": DS_LABEL, "gens": {g: GEN_LABEL[g] for g in ALL_GENS}, "figures": figs,
            "generated": time.strftime("%Y-%m-%d %H:%M"), "n_runs": int(len(df.groupby(["method", "dataset", "split", "seed"])))}


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PSyGenTAB — results review</title>
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f4f3f0;--text-primary:#0b0b0b;--text-secondary:#52514e;
--text-muted:#8c8b86;--grid:#e6e5e1;--border:#d9d8d3;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ref:#8c8b86}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--surface-2:#242422;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e88;--grid:#33332f;--border:#45443f;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ref:#8f8e88}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--text-muted:#8f8e88;--grid:#33332f;--border:#45443f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ref:#8f8e88}
body{margin:0}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh}
.wrap{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 6px}
.sub{color:var(--text-secondary);margin:0 0 14px}.muted{color:var(--text-muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.card{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.card .k{font-size:12px;color:var(--text-secondary)}.card .v{font-size:22px;font-weight:600;margin-top:2px}
.card .d{font-size:12px;color:var(--text-secondary)}
.filters{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin:8px 0 10px}
.filters label{display:flex;flex-direction:column;font-size:12px;color:var(--text-secondary);gap:4px}
select{font:inherit;color:var(--text-primary);background:var(--surface-1);border:1px solid var(--border);border-radius:6px;padding:5px 8px}
.chart{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:8px;position:relative}
svg text{fill:var(--text-secondary);font-size:11px}svg .axis-title{fill:var(--text-primary);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--text-secondary);margin:6px 4px 2px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.tip{position:fixed;pointer-events:none;background:var(--surface-2);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:10;max-width:320px}
.tip b{color:var(--text-primary)}.tip .r{color:var(--text-secondary)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{cursor:pointer;color:var(--text-secondary);font-weight:600;position:sticky;top:0;background:var(--surface-1)}
td:first-child,th:first-child{text-align:left}tr.ours td:first-child{font-weight:650}
.tablebox{max-height:560px;overflow:auto;border:1px solid var(--border);border-radius:10px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.gallery figure{margin:0;background:#fff;border:1px solid var(--border);border-radius:10px;padding:8px}
.gallery img{width:100%;display:block}.gallery figcaption{font-size:12px;color:#52514e;margin-top:4px}
.toggle{float:right}button{font:inherit;color:var(--text-primary);background:var(--surface-2);border:1px solid var(--border);
border-radius:6px;padding:4px 10px;cursor:pointer}
.pending{font-size:12.5px;color:var(--text-secondary);background:var(--surface-2);border:1px dashed var(--border);border-radius:8px;padding:8px 12px}
</style></head>
<body><div class="viz-root"><div class="wrap">
<button class="toggle" id="theme">Toggle dark mode</button>
<h1>PSyGenTAB — results review</h1>
<p class="sub">Generated <span id="gen"></span> · <span id="nruns"></span> released datasets evaluated · means over runs (5 splits) per dataset; "All datasets" = mean of per-dataset means.</p>
<div class="pending" id="pending"></div>
<div class="cards" id="cards"></div>

<h2>Privacy–utility map</h2>
<p class="sub">Each point is one method. Arrows go from a raw generator to the same generator + PSyGenTAB. Ideal: top, and at the dashed privacy reference.</p>
<div class="filters">
<label>Dataset<select id="ds"></select></label>
<label>Privacy (x)<select id="xm"></select></label>
<label>Utility (y)<select id="ym"></select></label>
</div>
<div class="chart"><svg id="scatter" width="100%" height="460"></svg>
<div class="legend" id="leg1"></div></div>

<h2>Per-dataset effect of PSyGenTAB</h2>
<p class="sub">One generator at a time: raw vs. + PSyGenTAB (ALM) vs. + PSyGenTAB (min. intervention), per dataset, ±1 SD over runs.</p>
<div class="filters">
<label>Generator<select id="gsel"></select></label>
<label>Metric<select id="msel"></select></label>
</div>
<div class="chart"><svg id="dots" width="100%" height="420"></svg><div class="legend" id="leg2"></div></div>

<h2>Fairness &amp; diversity — does PSyGenTAB erase subgroups or rare patients?</h2>
<p class="sub">Per generator, mean over datasets (±1 SD across datasets). Subgroups are the levels of every categorical covariate with ≥30 real rows; datasets without categorical covariates (Breast Cancer, Liver, Parkinson's) only contribute to the diversity, minority-class and tail metrics. Dashed line = ideal value where one exists.</p>
<div class="filters"><label>Metric<select id="fsel"></select></label></div>
<div class="chart"><svg id="fair" width="100%" height="330"></svg><div class="legend" id="leg3"></div></div>

<h2>Leaderboard</h2>
<p class="sub">Average rank across datasets (1 = best). Utility rank: TSTR AUROC, macro-F1, α-precision. Privacy rank: P, distance-MIA, DOMIAS, attribute inference (AUCs ranked by distance from 0.5). Fairness rank: representation parity, minority-class ratio, worst-subgroup utility, rare-category coverage, entropy and tail preservation (ratios ranked by distance from 1). Overall = mean of utility and privacy ranks. Click a column to sort. This table is the text alternative to the charts.</p>
<div class="tablebox"><table id="board"></table></div>

<h2>Paper figures</h2>
<div class="gallery" id="gallery"></div>
</div><div class="tip" id="tip"></div></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const FAM = {base:{lab:'Raw generator',col:'--s3',shape:'circle'},psygentab:{lab:'+ PSyGenTAB (ALM)',col:'--s1',shape:'diamond'},
  psygentab_min:{lab:'+ PSyGenTAB (min. intervention)',col:'--s1',shape:'diamond',hollow:true},dp:{lab:'DP baseline',col:'--s2',shape:'square'}};
const fmt = (v,d=3) => v==null||isNaN(v) ? '—' : (+v).toFixed(d);
const tip = $('#tip');
function showTip(e, html){tip.innerHTML=html;tip.style.display='block';const x=e.clientX+14,y=e.clientY+14;
  tip.style.left=Math.min(x, innerWidth-tip.offsetWidth-8)+'px';tip.style.top=Math.min(y, innerHeight-tip.offsetHeight-8)+'px';}
function hideTip(){tip.style.display='none';}
document.addEventListener('keydown', e=>{if(e.key==='Escape')hideTip();});
$('#theme').onclick=()=>{const r=document.documentElement;const dark=r.dataset.theme==='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);
  r.dataset.theme=dark?'light':'dark';renderAll();};
$('#gen').textContent=D.generated;$('#nruns').textContent=D.n_runs;
$('#pending').innerHTML = D.pending.length ? '<b>Pending (AWS GPU lane):</b> '+D.pending.join(', ')+'. Their slots fill in automatically when the page is regenerated.' : 'All planned models have results.';

function opt(sel, items, val){sel.innerHTML=items.map(([k,l])=>`<option value="${k}">${l}</option>`).join('');if(val)sel.value=val;}
opt($('#ds'), [['__all__','All datasets']].concat(Object.entries(D.datasets)), '__all__');
const priv=['mia_domias_auc','mia_dist_auc','P','aia_risk','link_risk','share_closer_train'].filter(k=>D.metrics[k]);
const util=['tstr_catboost_auroc','tstr_catboost_f1_macro','alpha_precision','beta_recall','c2st_auc'].concat(D.fair).filter(k=>D.metrics[k]);
const refOf=k=>D.metrics[k].dir==='half'?0.5:(D.metrics[k].dir==='one'?1:null);
opt($('#fsel'), D.fair.map(k=>[k,D.metrics[k].label]), 'repr_parity');
opt($('#xm'), priv.map(k=>[k,D.metrics[k].label]), 'mia_domias_auc');
opt($('#ym'), util.map(k=>[k,D.metrics[k].label]), 'tstr_catboost_auroc');
opt($('#gsel'), Object.entries(D.gens), 'bayesian_network');
opt($('#msel'), Object.keys(D.metrics).map(k=>[k,D.metrics[k].label]), 'mia_domias_auc');
['#ds','#xm','#ym','#gsel','#msel','#fsel'].forEach(s=>$(s).onchange=renderAll);

function cellsFor(ds){
  if(ds!=='__all__') return D.cells.filter(c=>c.dataset===ds);
  const by={};D.cells.forEach(c=>{(by[c.method]=by[c.method]||[]).push(c);});
  return Object.entries(by).map(([m,arr])=>{const o={method:m,dataset:'__all__',n:arr.length};
    Object.keys(D.metrics).forEach(k=>{const v=arr.map(a=>a[k]).filter(x=>x!=null);o[k]=v.length?v.reduce((a,b)=>a+b,0)/v.length:null;
      o[k+'_sd']=v.length>1?Math.sqrt(v.reduce((a,b)=>a+(b-o[k])**2,0)/(v.length-1)):null;});return o;});
}
function scale(d0,d1,r0,r1){return v=>r0+(v-d0)/(d1-d0||1)*(r1-r0);}
function niceExt(vals,ref){let lo=Math.min(...vals),hi=Math.max(...vals);if(ref!=null){lo=Math.min(lo,ref);hi=Math.max(hi,ref);}
  const p=(hi-lo)*0.08||0.01;return[lo-p,hi+p];}
function ticks(lo,hi,n=6){const st=Math.pow(10,Math.floor(Math.log10((hi-lo)/n)));const m=[1,2,2.5,5,10].find(k=>(hi-lo)/(st*k)<=n)*st;
  const out=[];for(let v=Math.ceil(lo/m)*m;v<=hi+1e-12;v+=m)out.push(+v.toFixed(10));return out;}
function marker(shape,x,y,r,fill,stroke,hollow){
  const s=css('--surface-1'), f=hollow?s:fill, st=hollow?fill:s, sw=hollow?1.8:1.5;
  if(shape==='circle')return `<circle cx="${x}" cy="${y}" r="${r}" fill="${f}" stroke="${st}" stroke-width="${sw}"/>`;
  if(shape==='square')return `<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}" rx="1.5" fill="${f}" stroke="${st}" stroke-width="${sw}"/>`;
  return `<path d="M${x} ${y-r*1.25}L${x+r*1.15} ${y}L${x} ${y+r*1.25}L${x-r*1.15} ${y}Z" fill="${f}" stroke="${st}" stroke-width="${sw}"/>`;
}
function axes(svg,W,H,m,xs,ys,xt,yt,xl,yl){let g='';
  xt.forEach(t=>{const x=xs(t);g+=`<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H-m.b}" stroke="${css('--grid')}" stroke-width="1"/><text x="${x}" y="${H-m.b+16}" text-anchor="middle">${t}</text>`;});
  yt.forEach(t=>{const y=ys(t);g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" stroke="${css('--grid')}" stroke-width="1"/><text x="${m.l-8}" y="${y+4}" text-anchor="end">${t}</text>`;});
  g+=`<text class="axis-title" x="${(m.l+W-m.r)/2}" y="${H-6}" text-anchor="middle">${xl}</text>`;
  g+=`<text class="axis-title" transform="translate(14 ${(m.t+H-m.b)/2}) rotate(-90)" text-anchor="middle">${yl}</text>`;return g;}
function legend(el,fams){el.innerHTML=fams.map(f=>{const F=FAM[f];return `<span><svg width="14" height="14">${marker(F.shape,7,7,5,css(F.col),'',F.hollow)}</svg>${F.lab}</span>`;}).join('');}

function renderScatter(){
  const svg=$('#scatter'),W=svg.clientWidth||900,H=460,m={l:62,r:24,t:18,b:46};
  const ds=$('#ds').value,xm=$('#xm').value,ym=$('#ym').value,cells=cellsFor(ds).filter(c=>c[xm]!=null&&c[ym]!=null);
  if(!cells.length){svg.innerHTML=`<text x="20" y="40">No results for this selection yet.</text>`;return;}
  const xref=refOf(xm), yref=ym==='tstr_catboost_auroc'?D.trtr[ds]:refOf(ym);
  const [x0,x1]=niceExt(cells.map(c=>c[xm]),xref),[y0,y1]=niceExt(cells.map(c=>c[ym]),yref);
  const xs=scale(x0,x1,m.l,W-m.r),ys=scale(y0,y1,H-m.b,m.t);
  let g=axes(svg,W,H,m,xs,ys,ticks(x0,x1),ticks(y0,y1),D.metrics[xm].label+(xref!=null?` (${xref} = ideal)`:''),D.metrics[ym].label);
  if(xref!=null)g+=`<line x1="${xs(xref)}" x2="${xs(xref)}" y1="${m.t}" y2="${H-m.b}" stroke="${css('--ref')}" stroke-dasharray="4 3"/>`;
  if(yref!=null){g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${ys(yref)}" y2="${ys(yref)}" stroke="${css('--ref')}" stroke-dasharray="2 3"/>`;
    if(ym==='tstr_catboost_auroc')g+=`<text x="${W-m.r-4}" y="${ys(yref)-5}" text-anchor="end">trained on real data (TRTR)</text>`;}
  const byM=Object.fromEntries(cells.map(c=>[c.method,c]));
  cells.forEach(c=>{const md=D.methods[c.method];if(md&&md.family==='psygentab'&&byM[md.gen]){const a=byM[md.gen];
    g+=`<line x1="${xs(a[xm])}" y1="${ys(a[ym])}" x2="${xs(c[xm])}" y2="${ys(c[ym])}" stroke="${css('--ref')}" stroke-width="1.2" marker-end="url(#ah)"/>`;}});
  const order={base:0,dp:1,psygentab_min:2,psygentab:3};
  cells.sort((a,b)=>order[D.methods[a.method].family]-order[D.methods[b.method].family]).forEach(c=>{
    const md=D.methods[c.method],F=FAM[md.family];const x=xs(c[xm]),y=ys(c[ym]);
    g+=`<g class="pt" data-m="${c.method}">${marker(F.shape,x,y,5.5,css(F.col),'',F.hollow)}<circle cx="${x}" cy="${y}" r="12" fill="transparent"/></g>`;
    if(md.family==='psygentab'||md.family==='dp')g+=`<text x="${x+8}" y="${y-7}">${md.family==='dp'?md.label.replace('PATE-GAN','PATE').replace('DP-CTAB-GAN+','DP-CTAB'):D.gens[md.gen]}</text>`;});
  svg.innerHTML=`<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${css('--ref')}"/></marker></defs>`+g;
  svg.querySelectorAll('.pt').forEach(el=>{const c=byM[el.dataset.m];el.onmousemove=e=>showTip(e,
    `<b>${D.methods[c.method].label}</b><br><span class="r">${ds==='__all__'?'mean over '+c.n+' datasets':D.datasets[ds]+', '+c.n+' runs'}</span><br>`+
    `${D.metrics[xm].label}: <b>${fmt(c[xm])}</b>${c[xm+'_sd']!=null?' ± '+fmt(c[xm+'_sd']):''}<br>${D.metrics[ym].label}: <b>${fmt(c[ym])}</b>${c[ym+'_sd']!=null?' ± '+fmt(c[ym+'_sd']):''}`);el.onmouseleave=hideTip;});
  legend($('#leg1'),['base','psygentab','psygentab_min','dp'].filter(f=>cells.some(c=>D.methods[c.method].family===f)));
}

function renderDots(){
  const svg=$('#dots'),W=svg.clientWidth||900,H=420,m={l:120,r:24,t:14,b:46};
  const g0=$('#gsel').value,met=$('#msel').value,dsets=Object.keys(D.datasets);
  const vars=[['base',g0],['psygentab',g0+'+psygentab'],['psygentab_min',g0+'+psygentab_min']];
  const rows=D.cells.filter(c=>vars.some(v=>v[1]===c.method)&&c[met]!=null);
  if(!rows.length){svg.innerHTML=`<text x="20" y="40">${D.gens[g0]}: pending (AWS GPU run) — no results yet.</text>`;$('#leg2').innerHTML='';return;}
  const ref=refOf(met);
  const vals=rows.flatMap(c=>[c[met]-(c[met+'_sd']||0),c[met]+(c[met+'_sd']||0)]);const[x0,x1]=niceExt(vals,ref);
  const xs=scale(x0,x1,m.l,W-m.r),band=(H-m.t-m.b)/dsets.length;
  let g='';ticks(x0,x1).forEach(t=>{const x=xs(t);g+=`<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H-m.b}" stroke="${css('--grid')}"/><text x="${x}" y="${H-m.b+16}" text-anchor="middle">${t}</text>`;});
  if(ref!=null)g+=`<line x1="${xs(ref)}" x2="${xs(ref)}" y1="${m.t}" y2="${H-m.b}" stroke="${css('--ref')}" stroke-dasharray="4 3"/>`;
  g+=`<text class="axis-title" x="${(m.l+W-m.r)/2}" y="${H-6}" text-anchor="middle">${D.metrics[met].label}${ref!=null?` (${ref} = ideal)`:''}</text>`;
  dsets.forEach((d,i)=>{const yc=m.t+band*(i+0.5);g+=`<text x="${m.l-10}" y="${yc+4}" text-anchor="end">${D.datasets[d]}</text>`;
    vars.forEach(([f,mm],k)=>{const c=rows.find(r=>r.method===mm&&r.dataset===d);if(!c)return;const y=yc+(k-1)*band*0.26,x=xs(c[met]),F=FAM[f];
      if(c[met+'_sd']!=null)g+=`<line x1="${xs(c[met]-c[met+'_sd'])}" x2="${xs(c[met]+c[met+'_sd'])}" y1="${y}" y2="${y}" stroke="${css(F.col)}" stroke-width="1.5" opacity=".7"/>`;
      g+=`<g class="dp" data-k="${mm}|${d}">${marker(F.shape,x,y,4.8,css(F.col),'',F.hollow)}<rect x="${x-10}" y="${y-8}" width="20" height="16" fill="transparent"/></g>`;});});
  svg.innerHTML=g;
  svg.querySelectorAll('.dp').forEach(el=>{const[mm,d]=el.dataset.k.split('|');const c=rows.find(r=>r.method===mm&&r.dataset===d);
    el.onmousemove=e=>showTip(e,`<b>${D.methods[mm].label}</b><br><span class="r">${D.datasets[d]}, ${c.n} runs</span><br>${D.metrics[met].label}: <b>${fmt(c[met])}</b>${c[met+'_sd']!=null?' ± '+fmt(c[met+'_sd']):''}`);el.onmouseleave=hideTip;});
  legend($('#leg2'),['base','psygentab','psygentab_min']);
}

function renderFair(){
  const svg=$('#fair'),W=svg.clientWidth||900,H=330,m={l:130,r:24,t:14,b:46},met=$('#fsel').value,ref=refOf(met);
  const gens=Object.keys(D.gens),vars=[['base',''],['psygentab','+psygentab'],['psygentab_min','+psygentab_min']];
  const stat=mm=>{const v=D.cells.filter(c=>c.method===mm&&c[met]!=null).map(c=>c[met]);if(!v.length)return null;
    const mu=v.reduce((a,b)=>a+b,0)/v.length;return{mu,sd:v.length>1?Math.sqrt(v.reduce((a,b)=>a+(b-mu)**2,0)/(v.length-1)):0,n:v.length};};
  const S={};gens.forEach(g=>vars.forEach(([f,suf])=>{S[g+suf]=stat(g+suf);}));
  const all=Object.values(S).filter(Boolean);if(!all.length){svg.innerHTML='<text x="20" y="40">Not computed yet.</text>';return;}
  const[x0,x1]=niceExt(all.flatMap(s=>[s.mu-s.sd,s.mu+s.sd]),ref);const xs=scale(x0,x1,m.l,W-m.r),band=(H-m.t-m.b)/gens.length;
  let g='';ticks(x0,x1).forEach(t=>{const x=xs(t);g+=`<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H-m.b}" stroke="${css('--grid')}"/><text x="${x}" y="${H-m.b+16}" text-anchor="middle">${t}</text>`;});
  if(ref!=null)g+=`<line x1="${xs(ref)}" x2="${xs(ref)}" y1="${m.t}" y2="${H-m.b}" stroke="${css('--ref')}" stroke-dasharray="4 3"/>`;
  g+=`<text class="axis-title" x="${(m.l+W-m.r)/2}" y="${H-6}" text-anchor="middle">${D.metrics[met].label}${ref!=null?` (${ref} = ideal)`:''} — ${D.metrics[met].dir==='down'?'lower':'higher'} is better${ref!=null?' / closer to ideal':''}</text>`;
  gens.forEach((gn,i)=>{const yc=m.t+band*(i+0.5);g+=`<text x="${m.l-10}" y="${yc+4}" text-anchor="end">${D.gens[gn]}</text>`;
    if(!S[gn]){g+=`<text x="${(m.l+W-m.r)/2}" y="${yc+4}" text-anchor="middle" font-style="italic">pending (AWS GPU run)</text>`;return;}
    vars.forEach(([f,suf],k)=>{const s=S[gn+suf];if(!s)return;const y=yc+(k-1)*band*0.26,x=xs(s.mu),F=FAM[f];
      g+=`<line x1="${xs(s.mu-s.sd)}" x2="${xs(s.mu+s.sd)}" y1="${y}" y2="${y}" stroke="${css(F.col)}" stroke-width="1.5" opacity=".7"/>`;
      g+=`<g class="fp" data-k="${gn+suf}">${marker(F.shape,x,y,4.8,css(F.col),'',F.hollow)}<rect x="${x-10}" y="${y-8}" width="20" height="16" fill="transparent"/></g>`;});});
  svg.innerHTML=g;
  svg.querySelectorAll('.fp').forEach(el=>{const mm=el.dataset.k,s=S[mm];el.onmousemove=e=>showTip(e,`<b>${D.methods[mm].label}</b><br><span class="r">mean over ${s.n} datasets</span><br>${D.metrics[met].label}: <b>${fmt(s.mu)}</b> ± ${fmt(s.sd)}`);el.onmouseleave=hideTip;});
  legend($('#leg3'),['base','psygentab','psygentab_min']);
}

let sortKey='rank_overall',sortAsc=true;
function renderBoard(){
  const cols=[['label','Method'],['rank_overall','Overall rank'],['rank_utility','Utility rank'],['rank_privacy','Privacy rank'],['rank_fairness','Fairness rank'],
    ['tstr_catboost_auroc','TSTR AUROC'],['tstr_catboost_f1_macro','Macro-F1'],['alpha_precision','α-prec.'],['c2st_auc','C2ST AUC'],
    ['P','P'],['mia_domias_auc','DOMIAS'],['mia_dist_auc','Dist. MIA'],['aia_risk','AIA risk'],['repr_parity','Repr. parity'],
    ['subgroup_util_ratio_min','Worst-subgr. util.'],['rare_category_coverage','Rare-cat. cov.'],['tail_preservation','Tail pres.'],['js_mean','Mean JS'],['n_datasets','Datasets']].filter(([k])=>k==='label'||k==='n_datasets'||k.startsWith('rank')||D.metrics[k]);
  const rows=[...D.board].sort((a,b)=>{const x=a[sortKey],y=b[sortKey];return(typeof x==='string'?x.localeCompare(y):(x-y))*(sortAsc?1:-1);});
  $('#board').innerHTML='<thead><tr>'+cols.map(([k,l])=>`<th data-k="${k}">${l}${k===sortKey?(sortAsc?' ▲':' ▼'):''}</th>`).join('')+'</tr></thead><tbody>'+
    rows.map(r=>`<tr class="${r.family.startsWith('psygentab')?'ours':''}">`+cols.map(([k])=>`<td>${k==='label'?r.label:k==='n_datasets'?r[k]:fmt(r[k],k.startsWith('rank')?2:3)}</td>`).join('')+'</tr>').join('')+
    D.pending.map(p=>`<tr><td>${p}</td><td colspan="${cols.length-1}" class="muted" style="text-align:left">pending (AWS GPU run)</td></tr>`).join('')+'</tbody>';
  $('#board').querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortAsc=k===sortKey?!sortAsc:true;sortKey=k;renderBoard();});
}

function renderCards(){
  const b=Object.fromEntries(D.board.map(r=>[r.method,r]));
  const pairs=Object.keys(D.gens).filter(g=>b[g]&&b[g+'+psygentab']);
  const avg=(f)=>pairs.length?pairs.reduce((a,g)=>a+f(g),0)/pairs.length:NaN;
  const dDom=avg(g=>Math.abs(b[g].mia_domias_auc-.5)-Math.abs(b[g+'+psygentab'].mia_domias_auc-.5));
  const dDist=avg(g=>Math.abs(b[g].mia_dist_auc-.5)-Math.abs(b[g+'+psygentab'].mia_dist_auc-.5));
  const dU=avg(g=>b[g+'+psygentab'].tstr_catboost_auroc-b[g].tstr_catboost_auroc);
  const top=D.board[0];
  $('#cards').innerHTML=[
    ['Best method overall',top?top.label:'—','average rank '+(top?fmt(top.rank_overall,2):'—')],
    ['DOMIAS leakage removed',fmt(dDom,3),'mean reduction of |AUC−0.5| over '+pairs.length+' generators'],
    ['Distance-MIA leakage removed',fmt(dDist,3),'same, distance attack'],
    ['Utility change',(dU>=0?'+':'')+fmt(dU,3),'TSTR AUROC, PSyGenTAB − raw']].map(([k,v,d])=>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join('');
}
function renderGallery(){$('#gallery').innerHTML=D.figures.map(f=>`<figure><a href="../figures/${f}" target="_blank"><img src="../figures/${f}" alt="${f}"></a><figcaption>${f.replace('.png','')}</figcaption></figure>`).join('');}
function renderAll(){renderCards();renderScatter();renderDots();renderFair();renderBoard();}
renderAll();renderGallery();addEventListener('resize',()=>{renderScatter();renderDots();renderFair();});
</script></body></html>
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    payload = build_payload()
    path = os.path.join(OUT, "index.html")
    with open(path, "w") as f:
        f.write(HTML.replace("__DATA__", json.dumps(payload)))
    print("wrote", path)


if __name__ == "__main__":
    main()
