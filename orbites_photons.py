#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction 1 — Orbites des photons au voisinage des trous noirs (D=4).
Validation Python du pipeline Mathematica Orbites_Photons.nb.

Reference : section 6.2 du chapitre 6 (these de reference).
Stage S4 — LPHEAG, Universite Cadi Ayyad, Marrakech.
Auteur : Hamza Alioua | Encadrant : Dr. S. Iraoui | Mai 2026.
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
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
ORBIT_COLOR = '#8B4513'  

M = 1.0
OUTPUT_DIR = 'figures_orbites'
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


#  Figure 6.4 : Potentiel effectif (massif + photon) 
print('Fig 6.4')

def Veff_m(r, L):
    """Potentiel effectif particule massive : (1 - 2M/r)(1 + L^2/r^2)."""
    return (1 - 2*M/r) * (1 + L**2/r**2)

def Veff_p(r, L):
    """Potentiel effectif photon : (1 - 2M/r)(L^2/r^2)."""
    return (1 - 2*M/r) * (L**2/r**2)

LMass, LPhot = 4.2*M, 4.35*M

# Extrema analytiques (eq. 6.36)
disc   = 1.0 - 12*M**2/LMass**2
rInner = (LMass**2/(2*M)) * (1 - np.sqrt(disc))   # C_i : max
rOuter = (LMass**2/(2*M)) * (1 + np.sqrt(disc))   # C_s : min
VInner, VOuter = Veff_m(rInner, LMass), Veff_m(rOuter, LMass)
VPhMax = Veff_p(3*M, LPhot)

print(f'  r_- = {rInner:.4f} M (V = {VInner:.5f})')
print(f'  r_+ = {rOuter:.4f} M (V = {VOuter:.5f})')
print(f'  V_max photon = {VPhMax:.5f} a r = 3M')

# Niveaux d'energie
E2A  = VOuter + 0.30*(1.0 - VOuter)
E2B  = 1.0 + 0.45*(VInner - 1.0)
E2C  = VInner
E2D  = VInner + 0.025
E2pB = 0.60*VPhMax
E2pC = VPhMax
E2pD = 1.20*VPhMax


def turning_pts(E2, L, rlo=2.001, rhi=900.0, n=80000):
    """Racines de Veff_m(r) = E2 pour r > 2M."""
    rs = np.linspace(rlo, rhi, n)
    fv = E2 - Veff_m(rs, L)
    sc = np.where(np.diff(np.sign(fv)))[0]
    roots = []
    for i in sc:
        try:
            roots.append(brentq(lambda r: E2 - Veff_m(r, L), rs[i], rs[i+1]))
        except Exception:
            pass
    return sorted(roots)


tp_A = turning_pts(E2A, LMass)
if len(tp_A) >= 3:
    ri_val, rm_val = tp_A[1], tp_A[2]
elif len(tp_A) == 2:
    ri_val, rm_val = tp_A
else:
    ri_val, rm_val = rInner*1.1, rOuter*1.5

print(f'  r_i = {ri_val:.3f} M, r_m = {rm_val:.3f} M')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# Panneau massif : echelle log
x_log = np.logspace(np.log10(1.005), np.log10(500), 6000)
r_log = 2*M*x_log
ax1.plot(x_log, Veff_m(r_log, LMass), color=COL[1], lw=2.3)

# Ligne A : segment [r_i, r_m] uniquement
ax1.plot([ri_val/(2*M), rm_val/(2*M)], [E2A, E2A], color=COL[0], lw=1.4)
# Label A centre sur le segment en echelle log (moyenne geometrique)
xA_label = np.sqrt(ri_val/(2*M) * rm_val/(2*M))
ax1.text(xA_label, E2A + 0.004, 'A',
         fontsize=12, fontstyle='italic', color=COL[0], ha='center')

# Tirets verticaux delimitant r_i et r_m
y_bot = 0.866
ax1.plot([ri_val/(2*M), ri_val/(2*M)], [y_bot, E2A], 'k--', lw=0.9, alpha=0.75)
ax1.plot([rm_val/(2*M), rm_val/(2*M)], [y_bot, E2A], 'k--', lw=0.9, alpha=0.75)
ax1.text(ri_val/(2*M)*0.88, y_bot + 0.002, r'$r_i$',
         fontsize=11, fontstyle='italic', fontweight='bold')
ax1.text(rm_val/(2*M)*1.04, y_bot + 0.002, r'$r_m$',
         fontsize=11, fontstyle='italic', fontweight='bold')

# Lignes B, C, D : pleine largeur
for E2v, lbl in [(E2B, 'B'), (E2C, 'C'), (E2D, 'D')]:
    ax1.axhline(E2v, color=COL[0], lw=1.4)
    ax1.text(60, E2v + 0.003, lbl, fontsize=12, fontstyle='italic', color=COL[0])

ax1.axhline(1.0, color='gray', ls=':', lw=0.9)

ax1.plot(rInner/(2*M), VInner, 'o', color='black', ms=7, zorder=5)
ax1.plot(rOuter/(2*M), VOuter, 'o', color='black', ms=7, zorder=5)
ax1.text(rInner/(2*M)*0.68, VInner + 0.013, r'$C_i$',
         fontsize=14, fontweight='bold')
ax1.text(rOuter/(2*M)*1.05, VOuter - 0.013, r'$C_s$',
         fontsize=14, fontweight='bold')

ax1.set_xscale('log')
ax1.set(xlabel=r'$r/2M$', ylabel=r'$V_{\mathrm{eff}}$',
        xlim=(1, 500), ylim=(0.86, 1.10))
ax1.set_title(r'(a) Particule massive ($\varepsilon=1$), $L = 4.2\,M$')
ax1.set_xticks([2, 5, 10, 50, 100, 500])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
frame_ax(ax1)

# Panneau photon : echelle lineaire
x_lin = np.linspace(1.005, 10.0, 3000)
ax2.plot(x_lin, Veff_p(2*M*x_lin, LPhot), color=COL[1], lw=2.3)

for E2v, lbl in [(E2pB, 'B'), (E2pC, 'C'), (E2pD, 'D')]:
    ax2.axhline(E2v, color=COL[0], lw=1.4)
    ax2.text(7.5, E2v + 0.025, lbl, fontsize=12, fontstyle='italic', color=COL[0])

ax2.plot(1.5, VPhMax, 'o', color='black', ms=7, zorder=5)
ax2.text(1.7, VPhMax + 0.040, r'$C_i$', fontsize=14, fontweight='bold')

ax2.set(xlabel=r'$r/2M$', ylabel=r'$V_{\mathrm{eff}}$',
        xlim=(1, 10), ylim=(0.0, 1.0))
ax2.set_title(r'(b) Particule sans masse ($\varepsilon=0$), $L = 4.35\,M$')
ax2.set_xticks([2, 4, 6, 8, 10])
frame_ax(ax2)

save_fig(fig, 'fig64_potentiel_effectif')
plt.close()


# Figure 6.5 : Precession du periapside (30 revolutions) 
print('Fig 6.5')

aOrb, eOrb = 55.0*M, 0.8
MoverL2 = M/(aOrb*(1 - eOrb**2))
u0Orb   = M/(aOrb*(1 - eOrb))
DeltaPhi = 3*np.pi*(2*M)/(aOrb*(1 - eOrb**2))

print(f'  a={aOrb} M, e={eOrb}, DeltaPhi = {DeltaPhi:.5f} rad/orbite')


def binet_massive(phi, y):
    """Equation de Binet massive : u'' + u = M/L^2 + 3 M u^2."""
    u, w = y
    return [w, MoverL2 + 3*M*u**2 - u]


phi_max65 = 30.0*2*np.pi
sol65 = solve_ivp(binet_massive, [0, phi_max65], [u0Orb, 0.0],
                  method='DOP853', rtol=1e-11, atol=1e-13,
                  dense_output=True)

phi_tab65 = np.linspace(0, phi_max65, 60000)
u_tab65   = sol65.sol(phi_tab65)[0]
r_tab65   = 1.0/u_tab65

# Coordonnees en r/M (cf. note methodologique du PDF)
x_all65 = r_tab65*np.cos(phi_tab65)/M
y_all65 = r_tab65*np.sin(phi_tab65)/M

N_per_rev = 60000//30
panels65  = [(1, 'T'), (2, '2T'), (8, '8T'), (30, '30T')]


def make_panel65(ax, n_rev, label):
    n_max = min(N_per_rev*n_rev, len(x_all65))
    ax.plot(x_all65[:n_max], y_all65[:n_max], color=ORBIT_COLOR, lw=0.9)

    ax.add_patch(Circle((0, 0), 2.0, color='black', zorder=10))
    ax.text(0, -3.6, 'Trou noir', color='black', ha='center', va='top',
            fontsize=8, fontstyle='italic', zorder=11)

    ax.axhline(0, color='black', lw=0.6)
    ax.axvline(0, color='black', lw=0.6)

    if label == 'T':
        # Annotation pedagogique : vecteur radial + arc phi
        angle, r_vec, r_arc = np.pi/4, 14.0, 3.5
        ax.plot([0, r_vec*np.cos(angle)], [0, r_vec*np.sin(angle)],
                'k-', lw=0.8, ls='--')
        phi_arc = np.linspace(0, angle, 80)
        ax.plot(r_arc*np.cos(phi_arc), r_arc*np.sin(phi_arc), 'k-', lw=1.1)
        ax.text(7.0, 9.5, r'$\dfrac{r}{2M}$', fontsize=10, ha='center',
                va='center', fontweight='bold')
        ax.text(4.6, 1.2, r'$\varphi$', fontsize=13, fontweight='bold')
        ax.plot(r_vec*np.cos(angle), r_vec*np.sin(angle),
                'o', color=ORBIT_COLOR, ms=6, zorder=12)

    ax.text(24, 27, label, fontsize=14, fontstyle='italic',
            ha='center', va='center')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.set_xticks([-20, -10, 0, 10, 20, 30])
    ax.set_yticks([-20, -10, 0, 10, 20, 30])
    ax.grid(True, color='lightgray', ls='--', lw=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)
    frame_ax(ax)


fig, axes = plt.subplots(2, 2, figsize=(11, 11), constrained_layout=True)
for ax, (n_rev, lbl) in zip(axes.flatten(), panels65):
    make_panel65(ax, n_rev, lbl)

save_fig(fig, 'fig65_orbite_massive')
plt.close()


#Figure 6.6 : Trois regimes photoniques
print('Fig 6.6')

xi_c = np.sqrt(27.0)*M


def binet_photon(phi, y):
    """Equation de Binet photon : u'' + u = 3 M u^2."""
    u, w = y
    return [w, 3*M*u**2 - u]


def first_int_ph(xi, u):
    return 1.0/xi**2 - u**2 + 2*M*u**3


# Cas (a) — retour parabolique : xi = 1.30 xi_c
xi_par  = 1.30*xi_c
u_per_a = brentq(lambda u: first_int_ph(xi_par, u), 1e-3, 0.32)

def ev_asymptote(phi, y):
    return y[0] - 5e-4
ev_asymptote.terminal, ev_asymptote.direction = True, -1

sol_a = solve_ivp(binet_photon, [0, 20.0], [u_per_a, 0.0],
                  method='DOP853', rtol=1e-13, atol=1e-15,
                  events=ev_asymptote, dense_output=True, max_step=0.002)

phi_asy_a = float(sol_a.t[-1])
alpha_A   = -123.0*np.pi/180.0
print(f'  (a) r_per = {1/u_per_a:.4f} M, '
      f'deflexion = {2*phi_asy_a - np.pi:.4f} rad')

n_par   = 3000
phi_arr = np.linspace(0, phi_asy_a, n_par)
r_arr   = np.minimum(1.0/sol_a.sol(phi_arr)[0], 50*M)

# Construction symetrique : branche arriere + branche avant, puis inversion
phi_bwd = phi_arr[::-1]
r_bwd   = r_arr[::-1]
x_bwd   = r_bwd*np.cos(alpha_A - phi_bwd)/(2*M)
y_bwd   = r_bwd*np.sin(alpha_A - phi_bwd)/(2*M)

phi_fwd = phi_arr[1:]
r_fwd   = r_arr[1:]
x_fwd   = r_fwd*np.cos(alpha_A + phi_fwd)/(2*M)
y_fwd   = r_fwd*np.sin(alpha_A + phi_fwd)/(2*M)

x_a = np.concatenate([x_bwd, x_fwd])[::-1]
y_a = np.concatenate([y_bwd, y_fwd])[::-1]

# Cas (b) — orbite circulaire instable : xi = xi_c
r0_B    = 7.0*M
u_B0    = 1.0/r0_B
w_B0    = np.sqrt(first_int_ph(xi_c, u_B0))
alpha_B = np.arctan2(2.5, 2.2)

def ev_photon_sphere(phi, y):
    return y[0] - 1.02/(3*M)
ev_photon_sphere.terminal, ev_photon_sphere.direction = True, +1

sol_b = solve_ivp(binet_photon, [0, 6*np.pi], [u_B0, w_B0],
                  method='DOP853', rtol=1e-14, atol=1e-16,
                  events=ev_photon_sphere, dense_output=True, max_step=0.002)

phi_b = np.linspace(0, sol_b.t[-1], 8000)
r_b   = 1.0/sol_b.sol(phi_b)[0]
x_b   = r_b*np.cos(alpha_B + phi_b)/(2*M)
y_b   = r_b*np.sin(alpha_B + phi_b)/(2*M)

# Cas (c) — capture gravitationnelle : xi = 0.85 xi_c
xi_cap  = 0.85*xi_c
r0_C    = 7.0*M
u_C0    = 1.0/r0_C
w_C0    = np.sqrt(first_int_ph(xi_cap, u_C0))
alpha_C = np.pi/4

def ev_horizon(phi, y):
    return y[0] - 1.0/(1.01*2*M)
ev_horizon.terminal, ev_horizon.direction = True, +1

sol_c = solve_ivp(binet_photon, [0, 8.0], [u_C0, w_C0],
                  method='DOP853', rtol=1e-12, atol=1e-14,
                  events=ev_horizon, dense_output=True, max_step=0.005)

phi_c = np.linspace(0, sol_c.t[-1], 3000)
r_c   = 1.0/sol_c.sol(phi_c)[0]
x_c   = r_c*np.cos(alpha_C + phi_c)/(2*M)
y_c   = r_c*np.sin(alpha_C + phi_c)/(2*M)

XMN, XMX = -3.0, 5.0
YMN, YMX = -4.0, 3.0


def make_photon_panel66(ax, x_orb, y_orb, e_label, title):
    ax.plot(x_orb, y_orb, color=ORBIT_COLOR, lw=1.6, clip_on=True)
    ax.add_patch(Circle((0, 0), 1.0, color='black', zorder=10))
    ax.text(0, 0, 'Trou noir', color='white', ha='center', va='center',
            fontsize=7, fontstyle='italic', fontweight='bold', zorder=11)

    vis = ((x_orb >= XMN) & (x_orb <= XMX) &
           (y_orb >= YMN) & (y_orb <= YMX))
    if vis.any():
        k = int(np.where(vis)[0][0])
        px, py = float(x_orb[k]), float(y_orb[k])
    else:
        px, py = 3.0, 2.5

    ax.plot(px, py, 'o', color=COL[0], ms=7, zorder=12)
    lx = float(np.clip(px - 0.55, XMN + 0.3, XMX - 1.1))
    ly = float(np.clip(py + 0.40, YMN + 0.4, YMX - 0.2))
    ax.text(lx, ly, 'Photon', fontsize=10, color='black', fontfamily='serif')

    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.text(XMX - 0.15, YMN + 0.3, e_label, fontsize=11, fontstyle='italic',
            ha='right',
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec='gray', alpha=0.85))

    ax.set_xlim(XMN, XMX)
    ax.set_ylim(YMN, YMX)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    frame_ax(ax)


fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), constrained_layout=True)
make_photon_panel66(axes[0], x_a, y_a, r'$E^2 < V_{\max}$', '(a) Retour parabolique')
make_photon_panel66(axes[1], x_b, y_b, r'$E^2 = V_{\max}$', '(b) Orbite circulaire instable')
make_photon_panel66(axes[2], x_c, y_c, r'$E^2 > V_{\max}$', '(c) Capture gravitationnelle')

save_fig(fig, 'fig66_orbites_photons')
plt.close()


# Figure 6.7 : Diagramme de phase T_r - r_(0r) pour RN-AdS 
print('Fig 6.7')

QV   = 1.0
rcRN = np.sqrt(6.0)*QV
PcRN = 1.0/(96.0*np.pi*QV**2)
TcRN = np.sqrt(6.0)/(18.0*np.pi*QV)


def M_RN(rp, P):
    return rp/2 + QV**2/(2*rp) + (4*np.pi/3)*P*rp**3

def T_RN(rp, P):
    return (1.0/(4*np.pi*rp))*(1 - QV**2/rp**2 + 8*np.pi*P*rp**2)

def r0_RN(rp, P):
    Mv = M_RN(rp, P)
    disc = 9*Mv**2 - 8*QV**2
    return np.where(disc >= 0,
                    (3*Mv + np.sqrt(np.maximum(disc, 0)))/2,
                    np.nan)

def r_extremal(P):
    """r_+ tel que T = 0 : 8 pi P x^2 + x - Q^2 = 0 avec x = r_+^2."""
    a, b, c = 8*np.pi*P, 1.0, -QV**2
    return np.sqrt((-b + np.sqrt(b**2 - 4*a*c))/(2*a))


Mc67  = M_RN(rcRN, PcRN)
r0c67 = r0_RN(rcRN, PcRN)
print(f'  r_0c = {r0c67:.5f} (theo {np.sqrt(6)+2:.5f})')


def make_branch67(Pr):
    Pv = Pr*PcRN
    rp_lo = r_extremal(Pv)*1.002
    rp_arr = np.linspace(rp_lo, 8.0, 6000)
    Tr  = T_RN(rp_arr, Pv)/TcRN
    r0r = r0_RN(rp_arr, Pv)/r0c67
    valid = ~np.isnan(r0r) & (Tr > 0) & (r0r > 0)
    return r0r[valid], Tr[valid]


r0r_sub,  Tr_sub  = make_branch67(0.8)
r0r_crit, Tr_crit = make_branch67(1.0)
r0r_sup,  Tr_sup  = make_branch67(1.5)


def find_extrema_67(r0r_arr, Tr_arr):
    order = np.argsort(r0r_arr)
    rs, Ts = r0r_arr[order], Tr_arr[order]
    dT = np.diff(Ts)/np.diff(rs)
    sc = np.where(np.diff(np.sign(dT)))[0]
    sf = [k for k in sc if 0.5 < rs[k] < 4.5]
    if len(sf) >= 2:
        return ((float(rs[sf[0]]), float(Ts[sf[0]])),
                (float(rs[sf[1]]), float(Ts[sf[1]])))
    return None, None


ext_max, ext_min = find_extrema_67(r0r_sub, Tr_sub)
if ext_max and ext_min:
    r02, T02 = ext_max
    r01, T01 = ext_min
    Tcoex    = 0.5*(T02 + T01)
    print(f'  r_02 = {r02:.4f}, r_01 = {r01:.4f}, T_coex = {Tcoex:.4f}')

fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
ax.plot(r0r_sub,  Tr_sub,  color=COL[1],    lw=2.2)
ax.plot(r0r_crit, Tr_crit, color='#7F4F24', lw=2.2)
ax.plot(r0r_sup,  Tr_sup,  color=COL[0],    lw=2.2)

ax.axhline(1.0, color='black', lw=0.8, ls=(0, (18, 5, 3, 5)), alpha=0.6)

# Labels inline des branches
ax.text(3.5,  1.005, r'$P < P_c$', fontsize=12, fontstyle='italic',
        ha='center', va='center')
ax.text(2.5,  1.050, r'$P = P_c$', fontsize=12, fontstyle='italic',
        ha='center', va='center')
ax.text(0.85, 1.18,  r'$P > P_c$', fontsize=12, fontstyle='italic',
        ha='center', va='center')
ax.text(2, 1.004, 'Point critique', fontsize=10, fontstyle='italic',
        ha='center', va='bottom')

# Boite de Maxwell : tirets verticaux + horizontale en T_coex
if ext_max and ext_min:
    ax.plot([r02, r02], [0.80, Tcoex], 'k--', lw=0.7, alpha=0.7)
    ax.plot([r01, r01], [0.80, Tcoex], 'k--', lw=0.7, alpha=0.7)
    ax.text(r02, 0.810, r'$\tilde{r}_{\mathrm{ps},2}$', fontsize=11, fontstyle='italic',
            ha='center', va='bottom')
    ax.text(r01, 0.810, r'$\tilde{r}_{\mathrm{ps},1}$', fontsize=11, fontstyle='italic',
            ha='center', va='bottom')
    ax.plot([r02, r01], [Tcoex, Tcoex], color='gray', lw=0.6, ls=':', alpha=0.8)
    # Label T_coex : temperature de coexistence estimee par le milieu des
    # extrema locaux (proxy qualitatif ; cf. limitation discutee dans le rapport)
    ax.text((r02 + r01)/2, Tcoex + 0.010, r'$T_{\mathrm{coex}}$',
            fontsize=11, color='gray', ha='center', va='bottom')

ax.set(xlabel=r'$\tilde{r}_{\mathrm{ps}} = r_{\mathrm{ps}}/r_{\mathrm{ps},c}$', ylabel=r'$T_r = T/T_c$',
       xlim=(0, 5), ylim=(0.80, 1.22))
ax.set_title('Diagramme de phase $T_r$\u2013$\\tilde{r}_{\\mathrm{ps}}$ \u2014 '
             'Trou noir RN-AdS ($Q=1$)')
frame_ax(ax)
save_fig(fig, 'fig67_Tr_r0r_diagram')
plt.close()


# Figure 6.8 : Derniere orbite circulaire stable (ISCO) pour RN-AdS 
print('Fig 6.8 (ISCO)')

#  Metrique RN-AdS et derivees : f(r) = 1 - 2M/r + Q^2/r^2 + (8 pi P/3) r^2
def f_RN(r, Mv, P):
    return 1 - 2*Mv/r + QV**2/r**2 + (8*np.pi*P/3)*r**2

def fp_RN(r, Mv, P):
    return 2*Mv/r**2 - 2*QV**2/r**3 + (16*np.pi*P/3)*r

def fpp_RN(r, Mv, P):
    return -4*Mv/r**3 + 6*QV**2/r**4 + (16*np.pi*P/3)

def Veff_massive(r, L, Mv, P):
    """Potentiel effectif massif (epsilon=1) : f(r)(1 + L^2/r^2)."""
    return f_RN(r, Mv, P)*(1 + L**2/r**2)

def horizon_RN(Mv, P):
    """Plus grande racine de f(r)=0."""
    rs = np.linspace(1e-3, 80, 300000)
    v  = f_RN(rs, Mv, P)
    roots = [brentq(f_RN, rs[i], rs[i+1], args=(Mv, P))
             for i in range(len(rs)-1) if v[i]*v[i+1] < 0]
    return max(roots) if roots else 1e-3

def isco_eq(r, Mv, P):
    """Condition ISCO independante de L : r f f'' + 3 f f' - 2 r f'^2 = 0."""
    f, fp, fpp = f_RN(r, Mv, P), fp_RN(r, Mv, P), fpp_RN(r, Mv, P)
    return r*f*fpp + 3*f*fp - 2*r*fp**2

def r_isco(Mv, P):
    """Plus petite racine de la condition ISCO exterieure a l'horizon."""
    rp = horizon_RN(Mv, P)
    rs = np.linspace(rp*1.0002, 50, 400000)
    v  = np.array([isco_eq(r, Mv, P) for r in rs])
    roots = [brentq(isco_eq, rs[i], rs[i+1], args=(Mv, P))
             for i in range(len(rs)-1) if v[i]*v[i+1] < 0]
    roots = [x for x in roots if x > rp*1.0002]
    return (min(roots) if roots else np.nan), rp

def L_circular(r, Mv, P):
    """Moment angulaire d'une orbite circulaire de rayon r (Veff'=0)."""
    f, fp = f_RN(r, Mv, P), fp_RN(r, Mv, P)
    return np.sqrt(fp*r**3/(2*f - r*fp))

#  Verifications analytiques 
ri_schw, _ = r_isco(1.0, 0.0)            # Schwarzschild Q=0,P=0 -> 6M (QV vaut 1 ici : on force Q=0)
# Note : QV est global =1 ; pour les controles on passe par des fonctions locales
def _isco_check(Mv, Qc, P):
    fL  = lambda r: 1 - 2*Mv/r + Qc**2/r**2 + (8*np.pi*P/3)*r**2
    fpL = lambda r: 2*Mv/r**2 - 2*Qc**2/r**3 + (16*np.pi*P/3)*r
    fppL= lambda r: -4*Mv/r**3 + 6*Qc**2/r**4 + (16*np.pi*P/3)
    eqL = lambda r: r*fL(r)*fppL(r) + 3*fL(r)*fpL(r) - 2*r*fpL(r)**2
    rs = np.linspace(2.0001, 30, 300000); v = np.array([eqL(r) for r in rs])
    R = [brentq(eqL, rs[i], rs[i+1]) for i in range(len(rs)-1) if v[i]*v[i+1] < 0]
    return min(R) if R else np.nan
print(f'  Controle Schwarzschild (Q=0,P=0) : r_ISCO = {_isco_check(1.0,0.0,0.0):.4f} M  (theo 6)')
print(f'  Controle RN extremal   (Q=M,P=0) : r_ISCO = {_isco_check(1.0,1.0,0.0):.4f} M  (theo 4)')

#  Trou noir critique RN-AdS (Q=1) : r+ = sqrt6, P = Pc 
rp_crit = np.sqrt(6.0)
M_crit  = M_RN(rp_crit, PcRN)
ri_crit, rplus_crit = r_isco(M_crit, PcRN)
L_crit  = L_circular(ri_crit, M_crit, PcRN)
print(f'  RN-AdS critique (r+=sqrt6, P=Pc, M={M_crit:.4f}) : '
      f'r_ISCO = {ri_crit:.4f} (= {ri_crit/M_crit:.3f} M), L_ISCO = {L_crit:.4f}')

#  Tableau r_ISCO(r+, P_r) pour Q=1 
print('  Tableau r_ISCO (Q=1) :')
print('    r_+      P_r=0.8   P_r=1.0   P_r=1.2')
for rp in [np.sqrt(6.0), 3.0, 4.0]:
    row = f'    {rp:6.3f} '
    for Pr in (0.8, 1.0, 1.2):
        M = M_RN(rp, Pr*PcRN); ri, _ = r_isco(M, Pr*PcRN)
        row += f'  {ri:7.3f} '
    print(row)

#  Figure 6.8 : deux panneaux 
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)

# Panneau (a) : fusion des orbites circulaires a l'ISCO (trou noir critique)
rr = np.linspace(rplus_crit*1.01, 14.0, 3000)
for fac, ls, col, lab in [(1.10, '-',  COL[0], r'$L = 1{,}10\,L_{\mathrm{ISCO}}$'),
                          (1.00, '-',  'black', r'$L = L_{\mathrm{ISCO}}$'),
                          (0.90, '--', COL[1], r'$L = 0{,}90\,L_{\mathrm{ISCO}}$')]:
    axA.plot(rr, Veff_massive(rr, fac*L_crit, M_crit, PcRN), ls, color=col, lw=2.2, label=lab)
axA.plot(ri_crit, Veff_massive(ri_crit, L_crit, M_crit, PcRN), 'o',
         color='black', ms=8, zorder=6)
axA.annotate(r'$r_{\mathrm{ISCO}}$',
             xy=(ri_crit, Veff_massive(ri_crit, L_crit, M_crit, PcRN)),
             xytext=(ri_crit+1.4, Veff_massive(ri_crit, L_crit, M_crit, PcRN)+0.50),
             fontsize=13, arrowprops=dict(arrowstyle='->', lw=1.0))
axA.axvline(rplus_crit, color='gray', ls=':', lw=1.0)
axA.text(rplus_crit+0.06, 0.30, r'$r_+$', color='gray', fontsize=11)
axA.set(xlabel=r'$r$', ylabel=r'$V_{\mathrm{eff}}(r)$', xlim=(rplus_crit-0.3, 14))
axA.set_title(r'(a) Fusion des orbites circulaires a l'+'\u2019'+'ISCO'+'\n'
              r'(RN-AdS critique : $Q=1$, $P=P_c$, $r_+=\sqrt{6}$)')
axA.legend(loc='upper left', frameon=False)
frame_ax(axA)

# Panneau (b) : r_ISCO en fonction de r+ pour trois pressions reduites
rp_grid = np.linspace(1.6, 6.0, 36)
for Pr, col in [(0.8, COL[1]), (1.0, '#7F4F24'), (1.2, COL[0])]:
    ys = [r_isco(M_RN(rp, Pr*PcRN), Pr*PcRN)[0] for rp in rp_grid]
    axB.plot(rp_grid, ys, color=col, lw=2.2, label=rf'$P_r = {Pr:.1f}$')
axB.set(xlabel=r'$r_+$', ylabel=r'$r_{\mathrm{ISCO}}$')
axB.set_title(r'(b) Rayon de l'+'\u2019'+r'ISCO vs rayon d'+'\u2019'+r'horizon ($Q=1$)')
axB.legend(loc='upper left', frameon=False)
frame_ax(axB)

save_fig(fig, 'fig68_isco_rnads')
plt.close()


print('\nToutes les figures exportees dans ./figures_orbites/')