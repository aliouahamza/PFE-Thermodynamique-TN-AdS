#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermodynamique des Trous Noirs Anti-de Sitter
Etude thermodynamique complète - Calcul symbolique + Figures
Stage S4 - LPHEAG, Universite Cadi Ayyad, Marrakech
Auteur : Hamza Alioua | Encadrant : Dr. S. Iraoui | Mars 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.scale import FuncScale
from matplotlib.ticker import FixedLocator, FuncFormatter   
from scipy.optimize import fsolve, brentq
import warnings, os

warnings.filterwarnings('ignore')

rcParams.update({
    'figure.figsize': (8, 5.5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Computer Modern Roman', 'Times New Roman'],
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.0,
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'figure.constrained_layout.use': True,
})

COLORS = ['#2166AC', '#D6604D', '#4DAF4A', '#FF7F00', '#984EA3',
          '#A65628', '#E41A1C', '#377EB8']

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
if _trapz is None:
    raise ImportError("Neither numpy.trapezoid nor numpy.trapz found")

def _symlog_fwd(y):
    return np.sign(y) * np.log10(1.0 + np.abs(y))

def _symlog_inv(y):
    return np.sign(y) * (np.power(10.0, np.abs(y)) - 1.0)

def set_symlog_nb(ax):
    """Applique l'echelle symlog du notebook (axe y).
    Fix : FixedLocator + FuncFormatter pour eviter le chevauchement des labels."""
    ax.set_yscale(FuncScale(ax.yaxis, (_symlog_fwd, _symlog_inv)))

    ticks = [-1e5, -1e4, -1e3, -1e2, -10, 0, 10, 1e2, 1e3, 1e4, 1e5]
    ax.yaxis.set_major_locator(FixedLocator(ticks))

    def _fmt(x, _):
        if x == 0:
            return '$0$'
        exp = int(np.floor(np.log10(abs(x))))
        if exp == 1:                          # 10 → $10$
            return r'$10$' if x > 0 else r'$-10$'
        return (rf'$10^{{{exp}}}$' if x > 0 else rf'$-10^{{{exp}}}$')

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))

def save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"),
                format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"),
                format='png', bbox_inches='tight', dpi=300)
    print(f"  [OK] {name}.pdf + {name}.png")

def adaptive_mesh(rmin, rmax, divs, n_base=2000, n_dense=500, eps=0.005):
    """Maillage dense autour de chaque divergence de Cp, uniforme ailleurs."""
    delta = 0.5
    pts = set()
    for r in np.linspace(rmin, rmax, n_base):
        pts.add(r)
    for d in divs:
        for r in np.linspace(max(rmin, d - delta), d - eps, n_dense):
            pts.add(r)
        for r in np.linspace(d + eps, min(rmax, d + delta), n_dense):
            pts.add(r)
    return np.array(sorted(pts))

# PARTIE 0 : SCHWARZSCHILD CLASSIQUE (Lambda = 0)
print("=" * 70)
print("PARTIE 0 : Schwarzschild classique (Lambda = 0)")
print("=" * 70)

def T_schw_flat(rp):     return 1.0 / (4.0 * np.pi * rp)
def M_schw_flat(rp):     return rp / 2.0
def S_schw_flat(rp):     return np.pi * rp**2
def C_schw_flat(rp):     return -2.0 * np.pi * rp**2
def G_schw_flat(rp):     return rp / 4.0

rp = np.linspace(0.1, 15, 1201)

# Fig 0a : T(r+)
fig, ax = plt.subplots()
ax.plot(rp, T_schw_flat(rp), color=COLORS[0], lw=2.5,
        label=r'$T = \frac{1}{4\pi\, r_+}$')
ax.set(xlabel=r'$r_+$', ylabel=r'$T$', xlim=(0,15), ylim=(0,0.5))
ax.set_title(r'Temperature de Hawking - Schwarzschild classique ($\Lambda = 0$)')
ax.legend(loc='upper right', fontsize=13)
save_fig(fig, "fig00a_T_rp_flat"); plt.close()

# Fig 0b : M(r+)
fig, ax = plt.subplots()
ax.plot(rp, M_schw_flat(rp), color=COLORS[1], lw=2.5,
        label=r'$M = \frac{r_+}{2}$')
ax.set(xlabel=r'$r_+$', ylabel=r'$M$', xlim=(0,15))
ax.set_title(r'Masse ADM - Schwarzschild classique')
ax.legend(loc='upper left', fontsize=13)
save_fig(fig, "fig00b_M_rp_flat"); plt.close()

# Fig 0c : S(r+)
fig, ax = plt.subplots()
ax.plot(rp, S_schw_flat(rp), color=COLORS[2], lw=2.5,
        label=r'$S = \pi\, r_+^2$')
ax.set(xlabel=r'$r_+$', ylabel=r'$S$', xlim=(0,15))
ax.set_title(r'Entropie de Bekenstein-Hawking - Schwarzschild classique')
ax.legend(loc='upper left', fontsize=13)
save_fig(fig, "fig00c_S_rp_flat"); plt.close()

# Fig 0d : C(r+)
fig, ax = plt.subplots()
ax.plot(rp, C_schw_flat(rp), color=COLORS[3], lw=2.5,
        label=r'$C = -2\pi\, r_+^2$')
ax.axhline(0, color='black', lw=0.8)
ax.set(xlabel=r'$r_+$', ylabel=r'$C$', xlim=(0,15))
ax.set_title(r'Capacite calorifique - Schwarzschild classique ($C < 0$ partout)')
ax.legend(loc='lower left', fontsize=13)
save_fig(fig, "fig00d_C_rp_flat"); plt.close()

# Fig 0e : G(r+)
fig, ax = plt.subplots()
ax.plot(rp, G_schw_flat(rp), color=COLORS[4], lw=2.5,
        label=r'$G = M - TS = \frac{r_+}{4}$')
ax.set(xlabel=r'$r_+$', ylabel=r'$G$', xlim=(0,15), ylim=(0,4))
ax.set_title(r'Energie libre de Gibbs - Schwarzschild classique ($\Lambda = 0$)')
ax.legend(loc='upper left', fontsize=13)
ax.annotate(r'$G = r_+/4 > 0$ partout' + '\n' + r'$\Rightarrow$ Pas de transition de phase',
            xy=(7, 1.75), fontsize=11, color='gray', fontstyle='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
save_fig(fig, "fig00e_G_rp_flat"); plt.close()

print("  G = r+/4 > 0 partout. C < 0 partout.\n")

# PARTIE I : SCHWARZSCHILD-AdS
print("=" * 70)
print("PARTIE I : Schwarzschild-AdS")
print("=" * 70)

def T_schw_ads(rp, P):   return 1.0/(4.0*np.pi*rp) + 2.0*P*rp
def M_schw_ads(rp, P):   return rp/2.0 + 4.0*np.pi*P*rp**3/3.0
def S_schw_ads(rp):      return np.pi*rp**2
def G_schw_ads(rp, P):   return rp/4.0 - 2.0*np.pi*P*rp**3/3.0

def Cp_schw_ads(rp, P):
    T = T_schw_ads(rp, P)
    dTdr = -1.0/(4.0*np.pi*rp**2) + 2.0*P
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(dTdr) < 1e-15, np.nan, T*2.0*np.pi*rp/dTdr)

#   r_HP = sqrt(3/(8 pi P)),   T_HP = (1/pi) sqrt(8 pi P / 3) = sqrt(8 P /(3 pi))
def rHP_schw_ads(P):     return np.sqrt(3.0/(8.0*np.pi*P))
def THP_schw_ads(P):     return np.sqrt(8.0*P/(3.0*np.pi))  

pressions_schw = [0.0005, 0.001, 0.002, 0.004, 0.008]

# Fig 1 : T(r+) 
rp = np.linspace(0.05, 12, 1501)
fig, ax = plt.subplots()
for i, P in enumerate(pressions_schw):
    ax.plot(rp, T_schw_ads(rp, P), color=COLORS[i], label=f'$P = {P}$')
    # Minimum de T situe a r_min = (8 pi P)^(-1/2)
    rp_min = 1.0/np.sqrt(8.0*np.pi*P)
    ax.plot(rp_min, T_schw_ads(rp_min, P), 'o', color=COLORS[i], ms=5)
ax.set(xlabel=r'$r_+$', ylabel=r'$T$', xlim=(0,12), ylim=(0,0.25))
ax.set_title(r'$T(r_+)$ - Schwarzschild-AdS')
ax.legend(loc='upper left', fontsize=10)
save_fig(fig, "fig01_T_rp_schw"); plt.close()

# Fig 2 : G(T) - transition de Hawking-Page
rp_param = np.linspace(0.05, 60, 3001)
fig, ax = plt.subplots()
for i, P in enumerate(pressions_schw):
    ax.plot(T_schw_ads(rp_param, P), G_schw_ads(rp_param, P),
            color=COLORS[i], label=f'$P = {P}$')
ax.axhline(0, color='black', lw=0.8, ls='--')
for i, P in enumerate(pressions_schw):
    THP = T_schw_ads(rHP_schw_ads(P), P)  
    ax.plot(THP, 0.0, 'o', color=COLORS[i], ms=5, zorder=6,
            markeredgecolor='black', markeredgewidth=0.5)
ax.set(xlabel=r'$T$', ylabel=r'$G$', xlim=(0,0.25), ylim=(-2,2))
ax.set_title(r'$G(T)$ - Transition de Hawking-Page')
ax.legend(loc='upper right', fontsize=10)
save_fig(fig, "fig02_G_T_schw"); plt.close()

# Fig 3 : Cp Schwarzschild-AdS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(r'Capacite calorifique $C_P(r_+)$ - Schwarzschild-AdS',
             fontsize=15, fontweight='bold')

for i, P in enumerate(pressions_schw):
    rdiv = 1.0/np.sqrt(8.0*np.pi*P)
    eps = 0.003
    rL = np.linspace(0.1, rdiv - eps, 401)
    rR = np.linspace(rdiv + eps, 15.0, 601)
    lbl = f'$P = {P}$'

    ax1.plot(rL, Cp_schw_ads(rL, P), color=COLORS[i], label=lbl)
    ax1.plot(rR, Cp_schw_ads(rR, P), color=COLORS[i])
    ax1.axvline(rdiv, color=COLORS[i], lw=0.7, ls=':', alpha=0.5)

    ax2.plot(rL, Cp_schw_ads(rL, P), color=COLORS[i], label=lbl)
    ax2.plot(rR, Cp_schw_ads(rR, P), color=COLORS[i])
    ax2.axvline(rdiv, color=COLORS[i], lw=0.7, ls=':', alpha=0.5)

# (a) Echelle symlog 
ax1.set_ylim(-1e5, 1e5)
set_symlog_nb(ax1)
ax1.axhline(0, color='black', lw=0.8, ls='--')
ax1.set(xlabel=r'$r_+$', ylabel=r'$C_P$', xlim=(0,15))
ax1.set_title(r'(a) Echelle symlog')
ax1.legend(loc='lower right', fontsize=8)

ax2.axhline(0, color='black', lw=0.8, ls='--')
ax2.set(xlabel=r'$r_+$', ylabel=r'$C_P$', xlim=(0,15), ylim=(-50,50))
ax2.set_title(r'(b) Zoom lineaire')
ax2.legend(loc='lower right', fontsize=8)
ax2.annotate(r'GTN ($C_P > 0$, hors plage)', xy=(10,15), fontsize=10,
             color='gray', fontstyle='italic')
ax2.annotate(r'PTN ($C_P < 0$)', xy=(1.5,-25), fontsize=10,
             color='gray', fontstyle='italic')

save_fig(fig, "fig03_Cp_rp_schw"); plt.close()

# PARTIE II : REISSNER-NORDSTROM-AdS (Q = 1)
print("=" * 70)
print("PARTIE II : Reissner-Nordstrom-AdS (Q = 1)")
print("=" * 70)

# Point critique (analogie van der Waals) : r_c = sqrt(6) Q, P_c v_c / T_c = 3/8
Q = 1.0
rc = np.sqrt(6.0)*Q
Pc = 1.0/(96.0*np.pi*Q**2)
Tc = np.sqrt(6.0)/(18.0*np.pi*Q)
vc = 2.0*rc
print(f"  r_c={rc:.6f}, P_c={Pc:.6e}, T_c={Tc:.6e}")
print(f"  P_c v_c / T_c = {Pc*vc/Tc:.6f}\n")

def T_rn(rp, P):     return 1/(4*np.pi*rp) - Q**2/(4*np.pi*rp**3) + 2*P*rp
def M_rn(rp, P):     return rp/2 + Q**2/(2*rp) + 4*np.pi*P*rp**3/3
def S_rn(rp):        return np.pi*rp**2
def G_rn(rp, P):     return M_rn(rp,P) - T_rn(rp,P)*S_rn(rp)
def P_rn(rp, T):     return T/(2*rp) - 1/(8*np.pi*rp**2) + Q**2/(8*np.pi*rp**4)
def Phi_rn(rp):      return Q/rp

def Cp_rn(rp, P):
    Tval = T_rn(rp, P)
    dTdr = -1/(4*np.pi*rp**2) + 3*Q**2/(4*np.pi*rp**4) + 2*P
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(dTdr)<1e-15, np.nan, Tval*2*np.pi*rp/dTdr)

def find_Cp_divs(Pval):
    """Racines de dT/dr = 0 : divergences de Cp (notebook findCPdivs)."""
    disc = 1.0 - Pval/Pc
    if disc < 0: return []
    elif abs(disc) < 1e-10: return [np.sqrt(1.0/(16*np.pi*Pval))]
    else:
        rm = np.sqrt((1-np.sqrt(disc))/(16*np.pi*Pval))
        rpl = np.sqrt((1+np.sqrt(disc))/(16*np.pi*Pval))
        return sorted([r for r in [rm, rpl] if r > Q])

# Fig 4 : T(r+)
P_ratios_T = [0.5, 0.8, 1.0, 1.2, 1.5]
rp = np.linspace(0.3, 12, 1501)
fig, ax = plt.subplots()
for i, pr in enumerate(P_ratios_T):
    sty = {'linestyle':'--', 'linewidth':2.5} if pr==1.0 else {}
    ax.plot(rp, T_rn(rp, pr*Pc), color=COLORS[i], label=f'$P/P_c={pr}$', **sty)
ax.plot(rc, Tc, 'ko', ms=8, zorder=5, label='Point critique')
ax.set(xlabel=r'$r_+$', ylabel=r'$T$', xlim=(0,12), ylim=(0,0.12))
ax.set_title(r'$T(r_+)$ - Reissner-Nordstrom-AdS ($Q=1$)')
ax.legend(loc='upper left', fontsize=10)
save_fig(fig, "fig04_T_rp_rn"); plt.close()

# Fig 5 : isothermes P-v (analogie van der Waals)
T_ratios = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10]
v_arr = np.linspace(0.8, 18, 1501)
fig, ax = plt.subplots()
for i, tr in enumerate(T_ratios):
    sty = {'linestyle':'--', 'linewidth':2.5} if tr==1.0 else {}
    ax.plot(v_arr, P_rn(v_arr/2, tr*Tc), color=COLORS[i], label=f'$T/T_c={tr}$', **sty)
ax.plot(vc, Pc, 'ko', ms=10, zorder=5, label='Point critique')
ax.set(xlabel=r'$v = 2\,r_+$', ylabel=r'$P$', xlim=(0.5,18), ylim=(0,0.008))
ax.set_title(r'Diagramme $P$-$v$ - Analogie van der Waals (RN-AdS)')
ax.legend(loc='upper right', fontsize=9)
save_fig(fig, "fig05_PV_rn"); plt.close()

# Fig 6 : G(T) - queue d'hirondelle
print("  G(T) numerique ...")

def find_rp_branches(P_val, T_val):
    """Racines de T(r,P) = T_val via brentq."""
    def f(rp): return T_rn(rp, P_val) - T_val
    rp_scan = np.linspace(Q+0.01, 25, 8000)
    f_vals = f(rp_scan)
    roots = []
    for j in range(len(f_vals)-1):
        if f_vals[j]*f_vals[j+1] < 0:
            try: roots.append(brentq(f, rp_scan[j], rp_scan[j+1]))
            except: pass
    return sorted(roots)

def find_crossing_direct(P_val, r1_guess, r2_guess):
    """Resout {T(r1)=T(r2), G(r1)=G(r2)} (notebook findCrossingDirect)."""
    def eqs(x):
        r1, r2 = x
        return [T_rn(r1,P_val)-T_rn(r2,P_val), G_rn(r1,P_val)-G_rn(r2,P_val)]
    try:
        sol = fsolve(eqs, [r1_guess, r2_guess], full_output=True)
        r1, r2 = sol[0]; res = abs(sol[1]['fvec'][0]) + abs(sol[1]['fvec'][1])
        if r1 > Q+0.001 and r2 > r1+0.01 and res < 1e-9:
            return (T_rn(r1,P_val), G_rn(r1,P_val), r1, r2)
    except: pass
    return None

P_ratios_G = [0.5, 0.7, 0.85, 1.0, 1.15]
r_phys_min, r_phys_max = Q + 0.005, 22.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(r"Energie libre de Gibbs $G(T)$ - Tracage numerique (RN-AdS, $Q=1$)",
             fontsize=15, fontweight='bold')

_offsets = {0.5: (0.004, 0.035, 'left'),
            0.7: (-0.003, -0.040, 'center'),
            0.85: (0.005, 0.040, 'left')}

crossings = []
for i, pr in enumerate(P_ratios_G):
    P_val = pr * Pc
    lbl = f'$P/P_c={pr}$'
    if pr == 1.0: lbl += ' (crit.)'
    sty_s = {'linestyle': '--', 'linewidth': 2.5} if pr == 1.0 else {}
    sty_inst = {'linestyle': (0, (4, 2)), 'linewidth': 1.0, 'alpha': 0.6}

    divs = find_Cp_divs(P_val)
    if len(divs) == 2:
        # Trois branches : PTN (plein) / instable (tirete fin) / GTN (plein)
        r1d, r2d = divs
        rPTN  = np.linspace(r_phys_min,  r1d - 0.002, 500)
        rINST = np.linspace(r1d + 0.002, r2d - 0.002, 300)
        rGTN  = np.linspace(r2d + 0.002, r_phys_max,  700)
        for axx in (ax1, ax2):
            axx.plot(T_rn(rPTN, P_val),  G_rn(rPTN, P_val),  color=COLORS[i], label=lbl, **sty_s)
            axx.plot(T_rn(rINST, P_val), G_rn(rINST, P_val), color=COLORS[i], **sty_inst)
            axx.plot(T_rn(rGTN, P_val),  G_rn(rGTN, P_val),  color=COLORS[i], **sty_s)
        if pr < 1.0:
            cr = find_crossing_direct(P_val, 0.8*r1d, 1.2*r2d)
            if cr is not None:
                crossings.append((cr[0], cr[1], cr[2], cr[3], COLORS[i], pr))
                print(f"    P/Pc={pr}: T_coex={cr[0]:.5f}, G={cr[1]:.4f}, "
                      f"r1={cr[2]:.3f}, r2={cr[3]:.3f}")
    else:
        # P >= Pc : branche unique
        rAll = np.linspace(r_phys_min, r_phys_max, 1200)
        for axx in (ax1, ax2):
            axx.plot(T_rn(rAll, P_val), G_rn(rAll, P_val), color=COLORS[i], label=lbl, **sty_s)

# Annotations des points de coexistence (panneau zoom)
for (Tcr, Gcr, r1c, r2c, col, pr) in crossings:
    ax2.plot(Tcr, Gcr, 'o', color=col, ms=6, zorder=10,
             markeredgecolor='black', markeredgewidth=0.8)
    offx, offy, ha = _offsets.get(pr, (0.003, 0.030, 'left'))
    ax2.annotate(f'$r_1={r1c:.2f},\\; r_2={r2c:.2f}$',
                 xy=(Tcr, Gcr), xytext=(Tcr + offx, Gcr + offy),
                 fontsize=8, fontweight='bold', color=col, ha=ha, va='center',
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                           edgecolor=col, alpha=0.92))

ax1.set(xlabel=r'$T$', ylabel=r'$G$', xlim=(0,0.06), ylim=(-0.5,1.5))
ax1.set_title(r'(a) Vue globale')
ax1.legend(loc='upper right', fontsize=9)

ax2.set(xlabel=r'$T$', ylabel=r'$G$', xlim=(0.025,0.050), ylim=(0.70,1.05))
ax2.set_title(r"(b) Zoom - Queue d'hirondelle")
ax2.legend(loc='upper right', fontsize=9)
ax2.annotate(r'PTN $\to$ GTN', xy=(0.040,0.77), fontsize=10,
             color='gray', fontstyle='italic', ha='center')

save_fig(fig, "fig06_GT_swallowtail"); plt.close()

# Fig 7 : courbe de coexistence PTN/GTN  (methode de continuation)
print("  Courbe de coexistence ...")

def find_coex_robust(Pval, r1_prev=None, r2_prev=None):
    """Resout {T(r1)=T(r2), G(r1)=G(r2)} a P = Pval (Maxwell equal-area)."""
    def eqs(x):
        r1, r2 = x
        return [T_rn(r1,Pval)-T_rn(r2,Pval), G_rn(r1,Pval)-G_rn(r2,Pval)]
    guesses = ([(r1_prev, r2_prev)] if r1_prev else []) + [
        (0.4*rc, 2.5*rc), (0.3*rc, 2.0*rc), (0.5*rc, 3.0*rc), (0.35*rc, 2.8*rc)]
    best_sol, best_res = None, 1e10
    for r1g, r2g in guesses:
        try:
            sol = fsolve(eqs, [r1g, r2g], full_output=True)
            r1, r2 = sol[0]; fv = sol[1]['fvec']; res = abs(fv[0])+abs(fv[1])
            if r1>Q+0.01 and r2>r1+0.1 and r1<rc and r2>rc and res<1e-10:
                Tco = T_rn(r1, Pval)
                if Tco>0 and res<best_res:
                    best_sol = (Tco, Pval, r1, r2); best_res = res
        except: continue
    return best_sol

P_fracs = np.arange(0.98, 0.04 - 1e-9, -0.01)
coex_data = []; r1p, r2p = 0.85*rc, 1.15*rc
for frac in P_fracs:
    res = find_coex_robust(frac*Pc, r1p, r2p)
    if res:
        coex_data.append(res); r1p, r2p = res[2], res[3]
coex_data.append((Tc, Pc, rc, rc))
coex_data.sort(key=lambda x: x[0])
# Filtrage monotone (elimine les doublons numeriques)
filt = [coex_data[0]]
for d in coex_data[1:]:
    if d[0]>filt[-1][0] and d[1]>filt[-1][1]: filt.append(d)
coex_data = filt
T_coex = np.array([d[0] for d in coex_data])
P_coex = np.array([d[1] for d in coex_data])
print(f"    {len(coex_data)-1} points + critique.\n")

fig, ax = plt.subplots()
ax.plot(T_coex, P_coex, color=COLORS[0], lw=2.5, label='Courbe de coexistence')
ax.plot(Tc, Pc, 'ro', ms=10, zorder=5, label=r'Point critique $(T_c, P_c)$')
ax.set(xlabel=r'$T$', ylabel=r'$P$', xlim=(0.008, Tc*1.1), ylim=(0, Pc*1.15))
ax.set_title(r'Courbe de coexistence PTN/GTN - Construction de Maxwell')
ax.annotate('PTN\n(petit trou noir)', xy=(0.020, 0.0015), fontsize=10,
            fontweight='bold', color=COLORS[0], ha='center')
ax.annotate('GTN\n(grand trou noir)', xy=(0.040, 0.00035), fontsize=10,
            fontweight='bold', color=COLORS[1], ha='center')
ax.legend(loc='upper left', fontsize=11)
save_fig(fig, "fig07_coexistence"); plt.close()

# Fig 8 : construction de Maxwell a T/Tc = 0.9
T_mw = 0.9 * Tc

def mw_eqs(x):
    """Systeme Maxwell : P(r1)=P(r2) et aire nette entre isotherme et palier = 0."""
    r1, r2 = x
    P1 = P_rn(r1, T_mw)
    v_int = np.linspace(2*r1, 2*r2, 2000)
    P_int = P_rn(v_int/2, T_mw)
    area_net = _trapz(P_int - P1, v_int)
    return [P1 - P_rn(r2, T_mw), area_net]

# Guess initial issu de coex_data (continuation)
idx_nearest = int(np.argmin(np.abs(T_coex - T_mw)))
r1g0, r2g0 = coex_data[idx_nearest][2], coex_data[idx_nearest][3]
guesses_grid = [(r1g0, r2g0)] + [(r1g, r2g)
                for r1g in np.linspace(1.1, 2.0, 6)
                for r2g in np.linspace(3.0, 6.5, 8)]

best_sol_mw, best_res_mw = None, 1e10
for r1g, r2g in guesses_grid:
    try:
        sol = fsolve(mw_eqs, [r1g, r2g], full_output=True)
        r1s, r2s = sol[0]
        res = sum(abs(sol[1]['fvec']))
        if (r1s > Q + 0.01 and r2s > r1s + 0.5
                and r1s < rc and r2s > rc and res < 1e-8
                and res < best_res_mw):
            best_sol_mw = (r1s, r2s)
            best_res_mw = res
    except Exception:
        continue

if best_sol_mw is None:
    raise RuntimeError("Construction de Maxwell : aucune solution valide trouvee.")

r1_mw, r2_mw = best_sol_mw
P_mw = P_rn(r1_mw, T_mw)
v1_mw, v2_mw = 2*r1_mw, 2*r2_mw

# Verification : aire+ = aire- (egalite des aires de Maxwell)
v_fa = np.linspace(v1_mw, v2_mw, 5000)
P_fa = P_rn(v_fa/2, T_mw)
area_p = _trapz(np.maximum(P_fa-P_mw, 0), v_fa)
area_m = _trapz(np.maximum(P_mw-P_fa, 0), v_fa)
print(f"  Maxwell T/Tc=0.9: r1={r1_mw:.4f}, r2={r2_mw:.4f}, P={P_mw:.6e}")
print(f"    Aire+={area_p:.4e}, Aire-={area_m:.4e}, |dA|/A+={abs(area_p-area_m)/max(area_p,1e-20):.2e}\n")

v_iso = np.linspace(1.0, 16, 2000)
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(v_iso, P_rn(v_iso/2, T_mw), color=COLORS[0], lw=2.5,
        label=r'Isotherme $T/T_c = 0.9$')
ax.plot([v1_mw, v2_mw], [P_mw, P_mw], color='red', lw=2.5, ls='--',
        label=f'$P_{{\\rm Maxwell}} = {P_mw:.5f}$')

v_fill = np.linspace(v1_mw, v2_mw, 1000)
P_fill = P_rn(v_fill/2, T_mw)
ax.fill_between(v_fill, P_fill, P_mw, where=(P_fill >= P_mw),
                alpha=0.35, color='#2ca02c', label=f'Aire + = {area_p:.4e}')
ax.fill_between(v_fill, P_fill, P_mw, where=(P_fill < P_mw),
                alpha=0.35, color='#d62728', label=f'Aire $-$ = {area_m:.4e}')

ax.plot(v1_mw, P_mw, 's', color='darkgreen', ms=12, zorder=10,
        markeredgecolor='black', markeredgewidth=1.0)
ax.plot(v2_mw, P_mw, 's', color='darkred', ms=12, zorder=10,
        markeredgecolor='black', markeredgewidth=1.0)

ax.annotate(f'$r_1 = {r1_mw:.3f}$\n$v_1 = {v1_mw:.3f}$',
            xy=(v1_mw, P_mw), xytext=(v1_mw - 0.8, P_mw + 0.0015),
            fontsize=10, fontweight='bold', color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='darkgreen', alpha=0.95))

ax.annotate(f'$r_2 = {r2_mw:.3f}$\n$v_2 = {v2_mw:.3f}$',
            xy=(v2_mw, P_mw), xytext=(v2_mw + 0.8, P_mw + 0.0015),
            fontsize=10, fontweight='bold', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='darkred', alpha=0.95))

ax.plot(vc, Pc, 'ko', ms=8, zorder=5, label='Point critique')

txt = (f'Verification numerique :\n'
       f'Aire+ = {area_p:.4e}\n'
       f'Aire- = {area_m:.4e}\n'
       f'$|\\Delta A|/A^+ = {abs(area_p-area_m)/max(area_p,1e-20):.1e}$')
ax.text(0.98, 0.50, txt, transform=ax.transAxes, fontsize=9,
        va='center', ha='right',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.95, ec='gray'))

ax.set(xlabel=r'$v = 2\,r_+$', ylabel=r'$P$', xlim=(0.5,16), ylim=(0,0.006))
ax.set_title(r'Construction de Maxwell - $T/T_c = 0.9$')
ax.legend(loc='upper right', fontsize=8)
save_fig(fig, "fig08_maxwell"); plt.close()

# Fig 9 : Cp RN-AdS (jusqu'a 2 divergences selon P/Pc)
P_ratios_Cp = [0.5, 0.8, 1.0, 1.3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(r'Capacite calorifique $C_P(r_+)$ - Reissner-Nordstrom-AdS ($Q=1$)',
             fontsize=15, fontweight='bold')

for i, pr in enumerate(P_ratios_Cp):
    Pval = pr*Pc
    divs = find_Cp_divs(Pval)
    sty = {'linestyle':'--', 'linewidth':2.5} if pr==1.0 else {}
    lbl = f'$P/P_c = {pr}$'
    if pr==1.0: lbl += ' (crit.)'
    eps9 = 0.003

    if len(divs) == 2:
        # Trois branches : PTN, instable, GTN
        r1d, r2d = divs
        rS1 = np.linspace(Q+0.01,   r1d - eps9, 400)
        rS2 = np.linspace(r1d+eps9, r2d - eps9, 400)
        rS3 = np.linspace(r2d+eps9, 10.0,       500)
        for axx in (ax1, ax2):
            axx.plot(rS1, Cp_rn(rS1, Pval), color=COLORS[i], label=lbl, **sty)
            axx.plot(rS2, Cp_rn(rS2, Pval), color=COLORS[i], **sty)
            axx.plot(rS3, Cp_rn(rS3, Pval), color=COLORS[i], **sty)
    elif len(divs) == 1:
        # Point critique : une seule divergence (second ordre)
        rS1 = np.linspace(Q+0.01,        divs[0] - eps9, 400)
        rS2 = np.linspace(divs[0]+eps9,  10.0,           500)
        for axx in (ax1, ax2):
            axx.plot(rS1, Cp_rn(rS1, Pval), color=COLORS[i], label=lbl, **sty)
            axx.plot(rS2, Cp_rn(rS2, Pval), color=COLORS[i], **sty)
    else:
        # P > Pc : pas de divergence, branche unique
        rAll = np.linspace(Q+0.01, 10.0, 800)
        for axx in (ax1, ax2):
            axx.plot(rAll, Cp_rn(rAll, Pval), color=COLORS[i], label=lbl, **sty)

    for d in divs:
        ax1.axvline(d, color=COLORS[i], lw=0.8, ls=':', alpha=0.5)
        ax2.axvline(d, color=COLORS[i], lw=0.8, ls=':', alpha=0.5)

# (a) symlog 
set_symlog_nb(ax1)
ax1.axhline(0, color='black', lw=0.8, ls='--')
ax1.set(xlabel=r'$r_+$', ylabel=r'$C_P$', xlim=(0.8,10), ylim=(-1e5, 1e5))
ax1.set_title(r'(a) $C_P(r_+)$ - Echelle symlog')
ax1.legend(loc='lower right', fontsize=8)

# (b) zoom lineaire
ax2.axhline(0, color='black', lw=0.8, ls='--')
ax2.set(xlabel=r'$r_+$', ylabel=r'$C_P$', xlim=(1.0,10), ylim=(-300,300))
ax2.set_title(r'(b) $C_P(r_+)$ - Zoom lineaire')
ax2.legend(loc='lower right', fontsize=8)
ax2.annotate('PTN', xy=(1.5,180), fontsize=11, fontweight='bold', color='gray', ha='center')
ax2.annotate('(instable)', xy=(3.0,-200), fontsize=10, color='gray', fontstyle='italic', ha='center')
ax2.annotate('GTN', xy=(8.5,180), fontsize=11, fontweight='bold', color='gray', ha='center')

save_fig(fig, "fig09_Cp_rn"); plt.close()

# Fig 10 : S(T) avec gap entropique dS = pi (r2^2 - r1^2)
fig, ax = plt.subplots()
P_ratios_ST = [0.5, 0.8, 1.0, 1.3]
for i, pr in enumerate(P_ratios_ST):
    Pval = pr*Pc
    sty = {'linestyle':'--', 'linewidth':2.5} if pr==1.0 else {'linewidth':2.0}
    sty_inst = {'linestyle': (0, (4, 2)), 'linewidth': 1.0, 'alpha': 0.5}
    divs = find_Cp_divs(Pval)
    if len(divs) == 2:
        r1d, r2d = divs
        rS1 = np.linspace(Q+0.005, r1d - 0.002, 400)
        rS2 = np.linspace(r1d+0.002, r2d - 0.002, 250)
        rS3 = np.linspace(r2d+0.002, 15.0, 500)
        ax.plot(T_rn(rS1, Pval), S_rn(rS1), color=COLORS[i], label=f'$P/P_c={pr}$', **sty)
        ax.plot(T_rn(rS2, Pval), S_rn(rS2), color=COLORS[i], **sty_inst)
        ax.plot(T_rn(rS3, Pval), S_rn(rS3), color=COLORS[i], **sty)
    else:
        rAll = np.linspace(Q+0.005, 15.0, 1000)
        ax.plot(T_rn(rAll, Pval), S_rn(rAll), color=COLORS[i], label=f'$P/P_c={pr}$', **sty)

# Gap entropique aux coexistences (P/Pc = 0.5 et 0.8), resolution EXACTE
for i_c, pr_c in enumerate([0.5, 0.8]):
    P_exact = pr_c * Pc
    idx_c = int(np.argmin(np.abs(P_coex - P_exact)))
    res_exact = find_coex_robust(P_exact, coex_data[idx_c][2], coex_data[idx_c][3])
    if res_exact is None:
        print(f"  [ATTENTION] Coexistence non trouvee a P/Pc={pr_c}")
        continue
    Tco, Pco, r1c, r2c = res_exact
    S1c, S2c = S_rn(r1c), S_rn(r2c); dSc = S2c - S1c
    print(f"  P/Pc={pr_c}: r1={r1c:.5f}, r2={r2c:.5f}, T_coex={Tco:.6f}, dS={dSc:.4f}")

    col = COLORS[i_c]
    ax.annotate('', xy=(Tco, S2c), xytext=(Tco, S1c),
                arrowprops=dict(arrowstyle='<->', color=col, lw=1.8, mutation_scale=12))
    y_axes = 0.97 - i_c * 0.18
    ax.annotate(r'$\Delta S = \pi(r_2^2-r_1^2)$' + f'\n$= {dSc:.1f}$  ($P/P_c={pr_c}$)',
                xy=(Tco, (S1c + S2c) / 2), xytext=(0.97, y_axes),
                xycoords='data', textcoords='axes fraction',
                fontsize=8, color=col, ha='right', va='top',
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, alpha=0.92))

ax.set(xlabel=r'$T$', ylabel=r'$S$', xlim=(0,0.08), ylim=(0,200))
ax.set_title(r'$S(T)$ - Reissner-Nordstrom-AdS ($Q=1$)')
ax.legend(loc='upper left', fontsize=10)
save_fig(fig, "fig10_ST_rn"); plt.close()

# Fig 11 : diagramme de phase P-T
fig, ax = plt.subplots()
P_full = np.concatenate([[0], P_coex, [0.004]])
T_full = np.concatenate([[0], T_coex, [Tc]])
ax.fill_betweenx(P_full, 0, T_full, color='blue', alpha=0.12, zorder=0)
ax.fill_betweenx(P_full, T_full, 0.085, color='red', alpha=0.12, zorder=0)
ax.plot(T_coex, P_coex, color=COLORS[0], lw=2.5, label='Ligne de coexistence', zorder=2)
ax.plot(Tc, Pc, 'ro', ms=10, zorder=5, label=r'Point critique $C$')
ax.annotate('Phase PTN\n(petit trou noir)', xy=(0.018,0.002), fontsize=12,
            fontweight='bold', color=COLORS[0], ha='center')
ax.annotate('Phase GTN\n(grand trou noir)', xy=(0.065,0.002), fontsize=12,
            fontweight='bold', color=COLORS[1], ha='center')
ax.annotate(r'$C\,(T_c=%.4f,\,P_c=%.4f)$' % (Tc, Pc),
            xy=(Tc, Pc), xytext=(Tc-0.008, Pc+0.0004), fontsize=9, fontweight='bold',
            color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax.set(xlabel=r'$T$', ylabel=r'$P$', xlim=(0,0.085), ylim=(0,0.004))
ax.set_title(r'Diagramme de phase $P$-$T$ - RN-AdS ($Q=1$)')
ax.legend(loc='upper left', fontsize=11)
save_fig(fig, "fig11_PT_phase"); plt.close()

# Fig 12 : potentiel electrique Phi(r+)
rp_phi = np.linspace(Q+0.01, 10, 1000)
fig, ax = plt.subplots()
ax.plot(rp_phi, Phi_rn(rp_phi), color=COLORS[4], lw=2.5, label=r'$\Phi = Q/r_+$')
ax.axhline(1.0, color='gray', lw=0.8, ls=':', label=r'$\Phi=1$ (extremal)')
ax.set(xlabel=r'$r_+$', ylabel=r'$\Phi$', xlim=(0,10), ylim=(0,1.2))
ax.set_title(r'Potentiel electrique $\Phi(r_+)$ - RN-AdS ($Q=1$)')
ax.legend(loc='upper right', fontsize=12)
save_fig(fig, "fig12_Phi_rp_rn"); plt.close()

# Fig 13 : M(r+) - enthalpie
rp_m = np.linspace(Q+0.01, 8, 1000)
fig, ax = plt.subplots()
for i, pr in enumerate([0.5, 1.0, 1.5]):
    sty = {'linestyle':'--'} if pr==1.0 else {}
    ax.plot(rp_m, M_rn(rp_m, pr*Pc), color=COLORS[i], label=f'$P/P_c={pr}$', **sty)
ax.set(xlabel=r'$r_+$', ylabel=r'$M$ (Enthalpie)', xlim=(0,8))
ax.set_title(r'$M(r_+)$ - RN-AdS ($Q=1$)')
ax.legend(loc='upper left', fontsize=11)
save_fig(fig, "fig13_M_rp_rn"); plt.close()

print("\n" + "="*70)
print("TOUTES LES FIGURES EXPORTEES (PDF + PNG) dans ./figures/")
print("="*70)