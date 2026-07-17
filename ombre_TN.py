#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction 2 — Ombre des trous noirs Schwarzschild-AdS et RN-AdS (D=4).
Validation Python du pipeline Mathematica Ombre_TN.nb.

Socle geometrique D=4 (coordonnees celestes, observable R_s) :
    Singh & Ghosh, Annals Phys. 395, 127 (2018), arXiv:1707.07125.
Extension au regime AdS charge (criticite van der Waals, R_s comme parametre d'ordre) :
    Wei & Liu, Phys. Rev. D 97, 104027 (2018), arXiv:1711.01522 ;
    Kubiznak-Mann, JHEP 1207, 033 (2012), arXiv:1205.0559.
Stage S4 — LPHEAG, Universite Cadi Ayyad, Marrakech.
Auteur : Hamza Alioua | Encadrant : Dr. S. Iraoui | Mai 2026.
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import quad
from scipy.optimize import brentq

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

rcParams.update({
    'figure.dpi'      : 150,
    'savefig.dpi'     : 300,
    'font.family'     : 'serif',
    'font.serif'      : ['Times New Roman', 'DejaVu Serif',
                         'Computer Modern Roman'],
    'mathtext.fontset': 'stix',
    'font.size'       : 13,
    'axes.labelsize'  : 14,
    'axes.titlesize'  : 13,
    'legend.fontsize' : 11,
    'xtick.labelsize' : 11,
    'ytick.labelsize' : 11,
    'lines.linewidth' : 2.0,
    'axes.grid'       : False,
    'axes.linewidth'  : 1.0,
})

COL = ['#2166AC', '#D6604D', '#4DAF4A', '#FF7F00',
       '#984EA3', '#A65628', '#E41A1C', '#377EB8']
BRUN_CUIR = (0.50, 0.31, 0.14)

M0 = 1.0
OUTPUT_DIR = 'figures_ombre'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.{ext}'),
                    format=ext, bbox_inches='tight', dpi=300)
    print(f'  [OK] {name}')

def frame_ax(ax, lw=1.0):
    for sp in ax.spines.values():
        sp.set_linewidth(lw)
        sp.set_color('black')
    ax.tick_params(direction='in', length=4, width=lw, top=True, right=True)

#  Metriques et observables

def f_SchAdS(r, M, ell):
    return 1.0 - 2.0*M/r + r**2/ell**2

def f_RNAdS(r, M, Q, P):
    return 1.0 - 2.0*M/r + Q**2/r**2 + (8.0*np.pi*P/3.0)*r**2

def rps_Sch(M):
    return 3.0 * M

def rps_RNAdS(M, Q):
    """r_ps = (3M + sqrt(9M^2 - 8Q^2))/2, independant de Lambda."""
    return 0.5 * (3.0*M + np.sqrt(np.maximum(9.0*M**2 - 8.0*Q**2, 0.0)))

def Rs_SchAdS(M, ell):
    rp = rps_Sch(M)
    return rp / np.sqrt(f_SchAdS(rp, M, ell))

def Rs_RNAdS(M, Q, P):
    rp = rps_RNAdS(M, Q)
    return rp / np.sqrt(f_RNAdS(rp, M, Q, P))

def M_SchAdS_of_rh(rh, ell):
    return 0.5 * rh * (1.0 + rh**2/ell**2)

def T_SchAdS_of_rh(rh, ell):
    return (1.0/(4.0*np.pi*rh)) * (1.0 + 3.0*rh**2/ell**2)

def M_RNAdS_of_rh(rh, P, Q=1.0):
    return 0.5*rh*(1.0 + Q**2/rh**2 + (8.0*np.pi*P/3.0)*rh**2)

def T_RNAdS_of_rh(rh, P, Q=1.0):
    return (1.0/(4.0*np.pi*rh)) * (1.0 - Q**2/rh**2 + 8.0*np.pi*P*rh**2)

def rh_extremal(P, Q=1.0):
    """Horizon extremal : T=0 <=> 8 pi P r_h^4 + r_h^2 - Q^2 = 0."""
    a, b, c = 8.0*np.pi*P, 1.0, -Q**2
    return np.sqrt((-b + np.sqrt(b**2 - 4*a*c))/(2*a))

# Constantes critiques RN-AdS (Q=1)
rcRN = np.sqrt(6.0)
PcRN = 1.0/(96.0*np.pi)
TcRN = np.sqrt(6.0)/(18.0*np.pi)
QV = 1.0

print('1. Verifications analytiques')
print(f'  r_c = sqrt(6)         = {rcRN:.6f}')
print(f'  P_c = 1/(96 pi)       = {PcRN:.6e}')
print(f'  T_c = sqrt(6)/(18 pi) = {TcRN:.6e}')
print(f'  P_c 2r_c / T_c        = {PcRN*2*rcRN/TcRN:.6f}  (theo 3/8)')

McRN = M_RNAdS_of_rh(rcRN, PcRN, 1.0)
print(f'  r_(ps,c) (2+sqrt(6))  = {rps_RNAdS(McRN, 1.0):.6f}  (theo {2+np.sqrt(6):.6f})')
print(f'  R_(s,c)               = {Rs_RNAdS(McRN, 1.0, PcRN):.6f}')


#  Figure 1 : Potentiel effectif Schwarzschild plat (D=4, M=0.5)
print('\nFig 7.1')

Mplat = 0.5
r_v71 = np.linspace(1.005, 4.0, 4000)
L_values = [1.0, 2.0, 3.0, 4.0, 5.0]
L_colors = [COL[0], COL[7], COL[2], COL[3], COL[1]]

fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
for L, c in zip(L_values, L_colors):
    V = (1.0 - 2.0*Mplat/r_v71) * L**2 / r_v71**2
    ax.plot(r_v71, V, color=c, lw=2.0, label=fr'$L = {L:.1f}$')

r_ps_norm = 1.5   # r_ps/(2M) = 3M/(2M) = 1.5
ax.axvline(r_ps_norm, color='gray', ls='--', lw=1.0, alpha=0.7)
ax.text(r_ps_norm + 0.05, 4.20, r'$r_{\mathrm{ps}} = 3M$',
        fontsize=12, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

for L in L_values:
    Vmax = L**2 / (27.0 * Mplat**2)
    if Vmax < 4.5:
        ax.plot(r_ps_norm, Vmax, 'o', color='black', ms=6, zorder=5)

ax.set(xlabel=r'$r/(2M)$', ylabel=r'$V_{\mathrm{eff}}(r)$',
       xlim=(1.0, 4.0), ylim=(0.0, 4.5))
ax.set_title(r'Potentiel effectif des photons — Schwarzschild plat ($D{=}4$, $M{=}0{,}5$)',
             fontsize=13)
ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=11)
frame_ax(ax)
save_fig(fig, 'fig1_Veff_photon_Sch')
plt.close()


#  Figure 2 : Ombres concentriques Schwarzschild
print('Fig 7.2')

M_values = [0.45, 0.60, 0.65, 0.70, 0.75]
M_colors = [COL[0], COL[7], COL[2], COL[3], COL[1]]
theta = np.linspace(0, 2*np.pi, 800)

fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
for Mv, c in zip(M_values, M_colors):
    Rs = np.sqrt(27.0) * Mv
    ax.plot(Rs*np.cos(theta), Rs*np.sin(theta), color=c, lw=2.0,
            label=fr'$M = {Mv:.2f}$')

ax.axhline(0, color='gray', lw=0.6, zorder=0)
ax.axvline(0, color='gray', lw=0.6, zorder=0)
ax.set(xlabel=r'$\alpha$', ylabel=r'$\beta$',
       xlim=(-4.5, 4.5), ylim=(-4.5, 4.5))
ax.set_aspect('equal')
ax.set_title(r'Ombres concentriques Schwarzschild ($D{=}4$, $\theta_o = \pi/2$)')
ax.legend(loc='center', framealpha=0.95, edgecolor='gray', fontsize=11)
frame_ax(ax)
save_fig(fig, 'fig2_ombres_Sch')
plt.close()


#  Figure 3 : R_s Schwarzschild-AdS, signature Hawking-Page 
print('Fig 7.3')

ell_fixe = 10.0
rh_min_phys = ell_fixe / np.sqrt(3.0)

rh_stab = np.linspace(rh_min_phys*1.001, 5.0*ell_fixe, 3000)
T_stab  = T_SchAdS_of_rh(rh_stab, ell_fixe) * ell_fixe
Rs_stab = Rs_SchAdS(M_SchAdS_of_rh(rh_stab, ell_fixe), ell_fixe) / ell_fixe

rh_ins = np.linspace(0.1*ell_fixe, rh_min_phys*0.999, 500)
T_ins  = T_SchAdS_of_rh(rh_ins, ell_fixe) * ell_fixe
Rs_ins = Rs_SchAdS(M_SchAdS_of_rh(rh_ins, ell_fixe), ell_fixe) / ell_fixe

T_HP_n  = 1.0/np.pi
T_min_n = np.sqrt(3.0)/(2.0*np.pi)
Rs_HP_n = 3.0 * np.sqrt(3.0/28.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

ax1.plot(T_stab, Rs_stab, color=COL[0], lw=2.5, label='Grande TN (stable)')
ax1.plot(T_ins,  Rs_ins,  color=COL[1], lw=1.8, ls='--', label='Petite TN (instable)')
ax1.axvline(T_HP_n,  color='gray', ls=(0, (5, 3)), lw=1.0, alpha=0.7)
ax1.axvline(T_min_n, color='gray', ls=(0, (5, 3)), lw=1.0, alpha=0.7)
ax1.plot(T_HP_n, Rs_HP_n, 'o', color=COL[2], ms=10, zorder=6,
         markeredgecolor='black', markeredgewidth=0.6)
ax1.text(T_HP_n + 0.035, 1.02, r'$T_{\mathrm{HP}}$',
         fontsize=11, fontstyle='italic',
         bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))
ax1.text(T_min_n - 0.030, 1.02, r'$T_{\min}$',
         fontsize=11, fontstyle='italic', ha='right',
         bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))
ax1.set(xlabel=r'$T\,\ell$', ylabel=r'$R_s/\ell$',
        xlim=(0.20, 1.0), ylim=(0.0, 1.1))
ax1.set_title(r'(a) $R_s(T)$ Schwarzschild-AdS, $\ell = 10$', fontsize=12)
ax1.legend(loc='center right', framealpha=0.95, edgecolor='gray', fontsize=10)
frame_ax(ax1)

ell_var = np.linspace(0.5, 30.0, 800)
P_var   = 3.0/(8.0*np.pi*ell_var**2)
Rs_var  = Rs_SchAdS(1.0, ell_var)

ax2.plot(P_var, Rs_var, color=COL[0], lw=2.5, label='Schwarzschild-AdS')
ax2.axhline(np.sqrt(27.0), color=COL[1], ls=(0, (5, 3)), lw=1.8,
            label=r'Limite plate : $R_s \to \sqrt{27}\,M$')
ax2.set(xlabel=r'$P$', ylabel=r'$R_s / M$',
        xlim=(0, 0.2), ylim=(0, 6.0))
ax2.set_title(r'(b) $R_s(P)$ Schwarzschild-AdS, $M = 1$', fontsize=12)
ax2.legend(loc='center right', framealpha=0.95, edgecolor='gray', fontsize=10)
frame_ax(ax2)

save_fig(fig, 'fig3_Rs_HP_SchAdS')
plt.close()


# Figure 4 : T(R_s) pour RN-AdS, avec coexistence de Maxwell
print("Fig 7.4")
def Rs_RN_of_rh(rh, P, Q=QV):
    return Rs_RNAdS(M_RNAdS_of_rh(rh, P, Q), Q, P)
def make_branch_RsT(Pr, n=4000):
    """
    Construit la courbe paramétrique T(R_s) à pression fixe.
    Retour :
        R_s en abscisse,
        T en ordonnée.
    """
    Pv = Pr * PcRN
    rh_lo = rh_extremal(Pv) * 1.0005
    rh_hi = max(12.0, 4.0 * np.sqrt(6.0) * QV)

    rh_arr = np.linspace(rh_lo, rh_hi, n)
    T_arr = T_RNAdS_of_rh(rh_arr, Pv)
    Rs_arr = Rs_RN_of_rh(rh_arr, Pv)
    valid = (
        (T_arr > 0)
        & np.isfinite(T_arr)
        & np.isfinite(Rs_arr)
        & (Rs_arr > 0)
    )
    return Rs_arr[valid], T_arr[valid]
def rh_max_dynamique(Pv, Q=QV):
    """
    Borne supérieure adaptative.

    r_(h,min-loc)^2 =
    (1 + sqrt(1 - 96*pi*P*Q^2))/(16*pi*P).
    """
    disc = 1.0 - 96.0 * np.pi * Pv * Q**2

    if disc <= 0:
        return 15.0

    rh_min_loc_th = np.sqrt(
        (1.0 + np.sqrt(disc)) / (16.0 * np.pi * Pv)
    )
    return max(15.0, 1.5 * rh_min_loc_th)
def maxwell_construction(Pr, Q=QV):
    """
    Construction de Maxwell dans le plan thermodynamique (T,S).
    La condition utilisée est :
        integral (T(r_h)-T_coex) dS = 0,
    avec dS = 2*pi*r_h dr_h.
    """
    Pv = Pr * PcRN
    rh_lo = rh_extremal(Pv) * 1.0005
    rh_hi = rh_max_dynamique(Pv, Q)
    n_test = max(5000, int(2000 * rh_hi / 10.0))
    rh_test = np.linspace(rh_lo, rh_hi, n_test)
    T_test = T_RNAdS_of_rh(rh_test, Pv, Q)
    dT = np.diff(T_test)
    sgn = np.sign(dT)
    sgn_changes = np.where(np.diff(sgn) != 0)[0]
    if len(sgn_changes) < 2:
        return None
    rh_max_loc = rh_test[sgn_changes[0] + 1]
    rh_min_loc = rh_test[sgn_changes[1] + 1]
    T_max_loc = T_RNAdS_of_rh(rh_max_loc, Pv, Q)
    T_min_loc = T_RNAdS_of_rh(rh_min_loc, Pv, Q)
    rh_far = max(rh_hi, 1.2 * rh_min_loc)
    def aire_balance(T_coex):
        try:
            r1 = brentq(
                lambda rh: (
                    T_RNAdS_of_rh(rh, Pv, Q) - T_coex
                ),
                rh_lo,
                rh_max_loc,
            )
            r3 = brentq(
                lambda rh: (
                    T_RNAdS_of_rh(rh, Pv, Q) - T_coex
                ),
                rh_min_loc,
                rh_far,
            )
            integral, _ = quad(
                lambda rh: (
                    T_RNAdS_of_rh(rh, Pv, Q) - T_coex
                )
                * 2.0
                * np.pi
                * rh,
                r1,
                r3,
                limit=100,
            )
            return integral
        except (ValueError, RuntimeError, OverflowError):
            return np.nan
    eps = (T_max_loc - T_min_loc) * 0.01
    try:
        T_coex = brentq(
            aire_balance,
            T_min_loc + eps,
            T_max_loc - eps,
        )
        return T_coex, rh_max_loc, rh_min_loc, rh_far
    except (ValueError, RuntimeError):
        return None
# Courbes avec R_s en abscisse et T en ordonnée
Rs_sub, T_sub = make_branch_RsT(0.6)
Rs_crt, T_crt = make_branch_RsT(1.0)
Rs_sup, T_sup = make_branch_RsT(1.5)
mx = maxwell_construction(0.6)
if mx is not None:
    T_coex, rh_max_loc, rh_min_loc, rh_far = mx
    Pv06 = 0.6 * PcRN
    rh1 = brentq(
        lambda rh: T_RNAdS_of_rh(rh, Pv06) - T_coex,
        rh_extremal(Pv06) * 1.001,
        rh_max_loc,
    )
    rh3 = brentq(
        lambda rh: T_RNAdS_of_rh(rh, Pv06) - T_coex,
        rh_min_loc,
        rh_far,
    )
    Rs_PTN = Rs_RN_of_rh(rh1, Pv06)
    Rs_GTN = Rs_RN_of_rh(rh3, Pv06)

    print(
        f"  T_coex   = {T_coex:.6f} "
        f"({T_coex / TcRN:.4f} T_c)"
    )
    print(
        f"  R_s(PTN) = {Rs_PTN:.5f}, "
        f"R_s(GTN) = {Rs_GTN:.5f}"
    )
    print(f"  Delta R_s = {Rs_GTN - Rs_PTN:.5f}")
else:
    print("  La construction de Maxwell a échoué.")
Rs_crit_value = Rs_RN_of_rh(rcRN, PcRN)
fig, ax = plt.subplots(
    figsize=(9.5, 6.5),
    constrained_layout=True,
)
ax.plot(
    Rs_sub,
    T_sub,
    color=COL[1],
    lw=2.3,
    label=r"$P = 0{,}6\,P_c$ (sous-critique)",
)
ax.plot(
    Rs_crt,
    T_crt,
    color=BRUN_CUIR,
    lw=2.3,
    label=r"$P = P_c$ (critique)",
)
ax.plot(
    Rs_sup,
    T_sup,
    color=COL[0],
    lw=2.3,
    label=r"$P = 1{,}5\,P_c$ (supercritique)",
)
# Température critique : ligne horizontale
ax.axhline(
    TcRN,
    color="gray",
    ls=(0, (5, 3)),
    lw=0.9,
    alpha=0.7,
)
ax.text(
    10.0,
    TcRN + 0.0015,
    r"$T_c$",
    fontsize=11,
    fontstyle="italic",
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.12",
        fc="white",
        ec="none",
    ),
)
ax.plot(
    Rs_crit_value,
    TcRN,
    "o",
    color="black",
    ms=10,
    zorder=6,
)
if mx is not None:
    # Palier horizontal T = T_coex
    ax.plot(
        [Rs_PTN, Rs_GTN],
        [T_coex, T_coex],
        color="black",
        ls=(0, (3, 3)),
        lw=1.0,
        alpha=0.9,
    )
    ax.plot(
        Rs_PTN,
        T_coex,
        "o",
        color=COL[2],
        ms=10,
        zorder=7,
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    ax.plot(
        Rs_GTN,
        T_coex,
        "o",
        color=COL[2],
        ms=10,
        zorder=7,
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    midpoint = 0.5 * (Rs_PTN + Rs_GTN)
    ax.text(
        midpoint,
        T_coex + 0.0012,
        r"$T_{\mathrm{coex}}$",
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.15",
            fc="white",
            ec="none",
        ),
    )
    ax.text(
        Rs_PTN,
        T_coex + 0.0028,
        "PTN",
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.12",
            fc="white",
            ec="none",
        ),
    )
    ax.text(
        Rs_GTN,
        T_coex + 0.0028,
        "GTN",
        fontsize=11,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.12",
            fc="white",
            ec="none",
        ),
    )
    # Saut horizontal Delta R_s
    arrow_y = T_coex - 0.0040
    ax.annotate(
        "",
        xy=(Rs_GTN - 0.15, arrow_y),
        xytext=(Rs_PTN + 0.15, arrow_y),
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            lw=1.3,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    ax.text(
        midpoint,
        arrow_y - 0.0025,
        r"$\Delta R_s$",
        fontsize=12,
        fontstyle="italic",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.15",
            fc="white",
            ec="none",
        ),
    )
ax.set(
    xlabel=r"$R_s$",
    ylabel=r"$T$",
    xlim=(3.5, 11.0),
    ylim=(0.005, 0.065),
)
ax.set_title(
    r"Diagramme $T(R_s)$ pour RN-AdS "
    r"$(Q = 1)$ à pressions fixes"
)
ax.legend(
    loc="upper left",
    framealpha=0.95,
    edgecolor="gray",
    fontsize=10,
)
frame_ax(ax)
save_fig(fig, "fig4_Rs_T_RNAdS")
plt.close(fig)


#  Figure 4bis : confirmation numerique de l'exposant critique beta = 1/2 (plan optique)
print('Fig 7.4bis (beta numerique)')

# Balayage isobarique sous-critique : pour chaque isobare, le saut Delta R_s a la
# coexistence de Maxwell est reporte en fonction de |t| = 1 - T_coex/T_c. La pente
# log-log proche-critique confirme Delta R_s ~ |t|^{1/2} (champ moyen, beta = 1/2).
Pr_beta = np.concatenate([np.linspace(0.45, 0.90, 16), np.linspace(0.91, 0.992, 12)])
t_beta, dRs_beta = [], []
for Pr in Pr_beta:
    mxb = maxwell_construction(Pr)
    if mxb is None:
        continue
    T_coex_b, rh_max_b, rh_min_b, rh_far_b = mxb
    Pv_b = Pr * PcRN
    try:
        rh1_b = brentq(lambda rh: T_RNAdS_of_rh(rh, Pv_b) - T_coex_b,
                       rh_extremal(Pv_b) * 1.001, rh_max_b)
        rh3_b = brentq(lambda rh: T_RNAdS_of_rh(rh, Pv_b) - T_coex_b,
                       rh_min_b, rh_far_b)
    except Exception:
        continue
    dRs_val = Rs_RN_of_rh(rh3_b, Pv_b) - Rs_RN_of_rh(rh1_b, Pv_b)
    t_val = 1.0 - T_coex_b / TcRN
    if t_val > 0 and dRs_val > 0:
        t_beta.append(t_val)
        dRs_beta.append(dRs_val)
t_beta = np.array(t_beta)
dRs_beta = np.array(dRs_beta)

# Ajustement log-log restreint a la fenetre proche-critique |t| <= t_cut
t_cut = 0.03
near = t_beta <= t_cut
beta_num, log_a = np.polyfit(np.log(t_beta[near]), np.log(dRs_beta[near]), 1)
print(f'  beta numerique (pente log-log, |t| <= {t_cut}) = {beta_num:.4f}  (theo 1/2)')

fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
ax.plot(t_beta[near], dRs_beta[near], 'o', color=COL[1], ms=7,
        markeredgecolor='black', markeredgewidth=0.5, zorder=5,
        label=r'points proche-critiques ($|t| \leq 0{,}03$)')
ax.plot(t_beta[~near], dRs_beta[~near], 'o', color='white', ms=7,
        markeredgecolor=COL[1], markeredgewidth=1.3, zorder=4,
        label='points hors régime asymptotique')
tt = np.linspace(0.8 * t_beta.min(), 1.1 * t_beta.max(), 200)
ax.plot(tt, np.exp(log_a) * tt**beta_num, '-', color='black', lw=1.8, zorder=3,
        label=fr'ajustement proche-critique : pente $= {beta_num:.3f}$')
i_anchor = int(np.argmin(t_beta))
C_half = dRs_beta[i_anchor] / np.sqrt(t_beta[i_anchor])
ax.plot(tt, C_half * np.sqrt(tt), '--', color=BRUN_CUIR, lw=1.8, zorder=2,
        label=r'pente $1/2$ (champ moyen)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(xlabel=r'$|t| = 1 - \widetilde{T}_{\mathrm{coex}}$',
       ylabel=r'$\Delta R_s = R_s^{\mathrm{GTN}} - R_s^{\mathrm{PTN}}$')
ax.set_title(r'Confirmation numérique de $\beta = 1/2$ : '
             r'$\Delta R_s \propto |t|^{1/2}$ (RN-AdS, $Q = 1$)')
ax.text(0.97, 0.06,
        r'$\beta_{\mathrm{num}} \simeq %.2f \;\to\; \beta_{\mathrm{theo}} = 1/2$' % beta_num,
        transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', fc='#fffbe6', ec='gray'))
ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray', fontsize=10)
frame_ax(ax)
ax.tick_params(which='both', direction='in', top=True, right=True)
save_fig(fig, 'fig4b_beta_loglog')
plt.close()


#  Figure 5 : Cartographie thermo-optique (P,T) -> R_s
print('Fig 7.5 (~20s)')

def Rs_of_TP_stable(Tv, Pv, Q=QV):
    try:
        rh_lo = rh_extremal(Pv, Q) * 1.0005
        rh_hi = max(rh_max_dynamique(Pv, Q), Tv/(1.5*Pv) + 5.0)
        rh_test = np.linspace(rh_lo, rh_hi, 300)
        T_test  = T_RNAdS_of_rh(rh_test, Pv, Q)
        residual = T_test - Tv
        roots = []
        for k in range(len(rh_test) - 1):
            if residual[k]*residual[k+1] < 0:
                try:
                    roots.append(brentq(
                        lambda rh: T_RNAdS_of_rh(rh, Pv, Q) - Tv,
                        rh_test[k], rh_test[k+1]))
                except Exception:
                    pass
        if not roots:
            return np.nan
        return float(np.mean([Rs_RN_of_rh(r, Pv, Q) for r in roots]))
    except Exception:
        return np.nan

T_grid = np.linspace(0.005, 0.080, 100)
P_grid = np.linspace(0.0005, 0.006, 100)
T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

Rs_mesh = np.full_like(T_mesh, np.nan)
for i in range(T_mesh.shape[0]):
    for j in range(T_mesh.shape[1]):
        Rs_mesh[i, j] = Rs_of_TP_stable(T_mesh[i, j], P_mesh[i, j])

# Construction de Maxwell etendue : Pr de 0.02 a 0.99
Pr_array = np.concatenate([np.linspace(0.02, 0.20, 25),
                           np.linspace(0.22, 0.99, 40)])
T_coex_arr, P_coex_arr = [], []
for Pr in Pr_array:
    res = maxwell_construction(Pr)
    if res is not None:
        T_coex_arr.append(res[0])
        P_coex_arr.append(Pr * PcRN)
T_coex_arr = np.array(T_coex_arr)
P_coex_arr = np.array(P_coex_arr)
print(f'  Coexistence : {len(T_coex_arr)} points, '
      f'P in [{P_coex_arr.min():.5f}, {P_coex_arr.max():.5f}]')

temperature_map = LinearSegmentedColormap.from_list(
    'TemperatureMap',
    ['#1F3F8C', '#4F7AC7', '#8DB4E3', '#D6E4F0',
     '#FAEEDC', '#F5C481', '#E89154', '#B22222'],
    N=256)

fig, ax = plt.subplots(figsize=(10.0, 7.2), constrained_layout=True)
levels = np.linspace(np.nanmin(Rs_mesh), np.nanmax(Rs_mesh), 60)
cs = ax.contourf(T_mesh, P_mesh, Rs_mesh, levels=levels,
                 cmap=temperature_map, extend='both')

if len(T_coex_arr) > 0:
    ax.plot(T_coex_arr, P_coex_arr, color='black', lw=2.8, zorder=4)

    # Label "Coexistence" pivote parallelement a la courbe en coordonnees ecran
    idx = len(T_coex_arr) // 5
    p1 = ax.transData.transform((T_coex_arr[idx-2], P_coex_arr[idx-2]))
    p2 = ax.transData.transform((T_coex_arr[idx+2], P_coex_arr[idx+2]))
    angle_deg = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))

    dT_loc = T_coex_arr[idx+2] - T_coex_arr[idx-2]
    dP_loc = P_coex_arr[idx+2] - P_coex_arr[idx-2]
    norm = np.hypot(dT_loc/0.075, dP_loc/0.0055)
    offT = +0.002 * (dP_loc/0.0055) / norm
    offP = -0.00015 * (dT_loc/0.075) / norm
    ax.text(T_coex_arr[idx] + offT, P_coex_arr[idx] + offP, 'Coexistence',
            fontsize=12, color='white', fontweight='bold',
            rotation=angle_deg, rotation_mode='anchor',
            ha='center', va='center')

ax.plot(TcRN, PcRN, 'o', color='red', ms=14, zorder=10,
        markeredgecolor='black', markeredgewidth=1.0)
ax.text(TcRN + 0.002, PcRN + 0.0002, r'$(T_c, P_c)$',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

ax.text(0.017, 0.0027, 'Phase PTN', fontsize=14, color='white',
        ha='center', va='center', fontweight='bold')
ax.text(0.055, 0.0015, 'Phase GTN', fontsize=14, color='black',
        ha='center', va='center', fontweight='bold')
ax.text(0.050, 0.0048, 'Phase supercritique', fontsize=14, color='white',
        ha='center', va='center', fontweight='bold')

cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$R_s$', fontsize=14, rotation=0, labelpad=15)
cbar.ax.tick_params(labelsize=11)

ax.set(xlabel=r'Température $T$', ylabel=r'Pression $P$',
       xlim=(0.005, 0.080), ylim=(0.0005, 0.006))
ax.set_title(r'Diagramme $(P, T)$ RN-AdS ($Q = 1$) avec carte de $R_s$')
frame_ax(ax)
save_fig(fig, 'fig5_PT_Rs_colormap')
plt.close()


#  Figure 6 : Taux d'emission energetique en D=4 
print('Fig 7.6')

def emission_rate(omega, T_H, Rs):
    """d^2E/(dw dt) = 2 pi^3 R_s^2 omega^3 / (exp(omega/T_H) - 1) en D=4."""
    return 2.0 * np.pi**3 * Rs**2 * omega**3 / (np.exp(omega/T_H) - 1.0)

omega_arr = np.arange(0.005, 0.5 + 0.001, 0.002)

# Cas (a) Schwarzschild plat M=1
T_a, Rs_a = 1.0/(8.0*np.pi*M0), np.sqrt(27.0)*M0
# Cas (b) Schwarzschild-AdS r_h=2, ell=10
T_b = T_SchAdS_of_rh(2.0, 10.0)
Rs_b = Rs_SchAdS(M_SchAdS_of_rh(2.0, 10.0), 10.0)
# Cas (c) RN-AdS r_h=2, Q=0.5, P=0.5 P_c(Q=0.5)
QV76 = 0.5
P_c76 = 0.5 / (96.0*np.pi*QV76**2)
T_c76 = T_RNAdS_of_rh(2.0, P_c76, QV76)
Rs_c76 = Rs_RNAdS(M_RNAdS_of_rh(2.0, P_c76, QV76), QV76, P_c76)

cases = [
    (T_a,   Rs_a,   r'Schwarzschild plat ($M = 1$)',                          COL[0]),
    (T_b,   Rs_b,   r'Schwarzschild-AdS ($r_+ = 2$, $\ell = 10$)',            COL[1]),
    (T_c76, Rs_c76, r'RN-AdS ($r_+ = 2$, $Q = 0{,}5$, $P = 0{,}5\,P_c$)',     COL[2]),
]

print(f'  Cas (a) : T_H = {T_a:.5f}, R_s = {Rs_a:.5f}')
print(f'  Cas (b) : T_H = {T_b:.5f}, R_s = {Rs_b:.5f}')
print(f'  Cas (c) : T_H = {T_c76:.5f}, R_s = {Rs_c76:.5f}')

fig, ax = plt.subplots(figsize=(9.5, 6.3), constrained_layout=True)
for T_H, Rs, label, c in cases:
    ax.plot(omega_arr, emission_rate(omega_arr, T_H, Rs),
            color=c, lw=2.3, label=label)

# Titre principal + sous-titre italique non superpose, via suptitle/title
fig.suptitle('Taux d' + '\u2019' + 'émission ($D{=}4$), '
             r'$\sigma_{\mathrm{lim}} = \pi R_s^2$',
             fontsize=13, y=1.02)
ax.set_title('(Spectre local dans l' + '\u2019' + 'approximation de corps noir)',
             fontsize=11, fontstyle='italic', pad=8)

ax.set(xlabel=r'Fréquence $\omega$', ylabel=r'$d^2 E / (d\omega\, dt)$',
       xlim=(0, 0.5), ylim=(0, 0.32))
ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=10)
frame_ax(ax)
save_fig(fig, 'fig6_emission_rate')
plt.close()

print('\nToutes les figures exportees dans ./figures_ombre/')