exec(open('diag_voi.py', encoding='utf8').read().split('# per patient choices')[0])
import numpy as np
for tagv in ('16x12','32x16','48x24'):
    rs = np.load(f'results/_diag_resid_{tagv}.npy')
    df['pred2'] = rs[df.row.values, df.blk.values]
    out=[]
    for row, q in df.groupby('row'):
        if len(q)<2: continue
        v = q.loc[q.pred2.idxmin()]
        out.append((v.real_u, v.real_flip, q.real_u.min()))
    o=np.array(out)
    print(tagv, 'spearman', round(spearmanr(df.pred2, df.real_u).statistic,3), 'VOI choice real_u', o[:,0].mean().round(4), 'flip', o[:,1].mean().round(4), 'oracle', o[:,2].mean().round(4))
