"""Fixed experimental hypotheses, not verified physical surface ground truth.

Early marker: first sufficiently prominent lobe in 960..1030. Template: polarity-
invariant normalized correlation to an equally specimen-sampled fold-train median
pulse. All thresholds below are frozen before CV; no previous holdout is accessed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks

from src.features.experimental_anchor_calculator import extension_at_anchor
from src.features.loading import load_measurement

VARIANTS=('legacy','early_legacy_windows','early_adapted','template_adapted')
SEARCH=(960.,1030.)
GRID=np.arange(-2.5,4.5001,.05)


def early_index(x,y):
    """First abs lobe ≥35% of search maximum, prominence ≥20%; no label use."""
    indices=np.flatnonzero((x>=SEARCH[0])&(x<=SEARCH[1]))
    if len(indices)<3: raise ValueError('early_search_missing')
    centered=y-np.median(y);amplitude=abs(centered[indices]);peak=amplitude.max()
    if not np.isfinite(peak) or peak<=0: raise ValueError('flat_signal')
    peaks,_=find_peaks(amplitude,height=.35*peak,prominence=.20*peak)
    if not len(peaks): raise ValueError('no_early_lobe')
    return int(indices[peaks[0]])


@dataclass
class PulseTemplate:
    values: np.ndarray
    member_ids: list


def fit_template(training_rows, per_specimen=12):
    """Training rows ONLY; equal deterministic counts per specimen, no labels."""
    pulses=[];ids=[]
    for _,group in training_rows.groupby('specimen_id',sort=True):
        group=group.sort_values('instance_id')
        for row in group.iloc[np.unique(np.linspace(0,len(group)-1,min(per_specimen,len(group)),dtype=int))].itertuples():
            wave=load_measurement(row.file_path);x,y=wave.x.to_numpy(),wave.y.to_numpy()
            try: index=early_index(x,y)
            except ValueError: continue
            if x[index]+GRID[0]<x.min() or x[index]+GRID[-1]>x.max(): continue
            pulse=np.interp(x[index]+GRID,x,y);pulse-=np.median(pulse)
            norm=np.linalg.norm(pulse)
            if norm<=0: continue
            pulse*=np.sign(pulse[np.argmin(abs(GRID))]) or 1
            pulses.append(pulse/norm);ids.append(row.instance_id)
    if len(pulses)<5: raise ValueError('Insufficient fold-training template pulses')
    template=np.median(pulses,axis=0);template-=template.mean();template/=np.linalg.norm(template)
    return PulseTemplate(template,ids)


def template_index(x,y,template):
    """Absolute NCC handles pulse polarity; amplitude gate rejects tiny matches."""
    grid=np.arange(x.min(),x.max()+.00001,.05);values=np.interp(grid,x,y)
    length=len(GRID);ones=np.ones(length)
    z=values-np.median(values)
    dot=correlate(z,template.values,mode='valid',method='fft')
    sums=correlate(z,ones,mode='valid',method='fft')
    energy=correlate(z*z,ones,mode='valid',method='fft')-sums*sums/length
    scores=np.abs(dot)/np.sqrt(np.maximum(energy,1e-20))
    centers=grid[:len(scores)]-GRID[0]
    valid=(centers>=SEARCH[0])&(centers<=SEARCH[1])
    maximum=np.max(abs(z[(grid>=SEARCH[0])&(grid<=SEARCH[1])]))
    valid &= abs(np.interp(centers,grid,z))>=.35*maximum
    scores[~valid]=-np.inf
    best=int(np.argmax(scores))
    if not np.isfinite(scores[best]) or scores[best]<.45: raise ValueError('low_template_ncc')
    return int(np.argmin(abs(x-centers[best]))),float(scores[best])


def adapted_start(x,y,index):
    """Guard the pulse packet: post start 6..9 units; no blind +3 relabeling."""
    t=x-x[index];dt=np.median(np.diff(x))
    envelope=gaussian_filter1d(abs(y-np.median(y)),sigma=max(.15/dt,.5))
    local=(t>=-2.5)&(t<=8)
    active=(t>=0)&(t<=8)&(envelope>=.20*np.max(envelope[local]))
    end=np.max(t[active]) if active.any() else 5.
    return float(np.clip(end+1,6,9))


def extract_variant(path,variant,extension_columns,template=None):
    """Failures retain every ID: NaN extension, reason in diagnostics, no fallback."""
    wave=load_measurement(path);x,y=wave.x.to_numpy(),wave.y.to_numpy()
    try:
        if not np.isfinite(wave.to_numpy()).all(): raise ValueError('nonfinite_signal')
        score=np.nan
        if variant=='template_adapted': index,score=template_index(x,y,template)
        else: index=early_index(x,y)
        adapted=variant.endswith('_adapted')
        start=adapted_start(x,y,index) if adapted else 3.
        noise=(-10.,-5.) if adapted else (70.,80.)
        if ((x-x[index]>=noise[0])&(x-x[index]<=noise[1])).sum()<5: raise ValueError('pre_noise_missing')
        features=extension_at_anchor(x,y,index,start,*noise)
        result=np.array([features[c] for c in extension_columns],dtype=float)
        result[~np.isfinite(result)]=np.nan
        return result,dict(status='ok',anchor_x=float(x[index]),post_start=start,ncc=score)
    except (ValueError,IndexError,ZeroDivisionError,FloatingPointError) as exc:
        return np.full(len(extension_columns),np.nan),dict(status=str(exc),anchor_x=np.nan,post_start=np.nan,ncc=np.nan)
