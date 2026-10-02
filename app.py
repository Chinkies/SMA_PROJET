"""
app.py - Interface Streamlit pour la simulation des systèmes de vote
et l'analyse du vote stratégique (Projet 4).
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Imports des modules du projet
from population import (
    CANDIDATS,
    calculer_optimum_social,
    fixer_aleatoire,
    generer_election_aleatoire,
    generer_population,
    generer_sondage,
)
from voting_systems import (
    auditer_election,
    calculer_social_welfare,
    vote_approval,
    vote_plurality,
    vote_stv,
    vote_two_round,
)

# Configuration de la page
st.set_page_config(
    page_title="Audit Systèmes de Vote",
    page_icon="🗳️",
    layout="wide"
)

# =============================================================================
# DÉFINITIONS DU CAS D'ÉCOLE SPATIAL PAR DÉFAUT
# =============================================================================
CANDIDATS_SPATIAUX_DEFAUT = [
    {"nom": "Bob", "x": -0.60, "y": 0.20, "couleur": "#1f77b4"},
    {"nom": "Raoul", "x": -0.70, "y": 0.40, "couleur": "#d62728"},
    {"nom": "Jackie", "x": 0.00, "y": 0.00, "couleur": "#2ca02c"},
    {"nom": "Mark", "x": 0.65, "y": -0.30, "couleur": "#ff7f0e"},
    {"nom": "Francis", "x": 0.75, "y": -0.60, "couleur": "#9467bd"},
]

CLUSTERS_SPATIAUX_DEFAUT = [
    {"nom": "Bloc Gauche", "x": -0.65, "y": 0.30, "sigma": 0.22, "poids": 35.0, "couleur": "#436CAE"},
    {"nom": "Bloc Centre", "x": 0.00, "y": 0.05, "sigma": 0.28, "poids": 20.0, "couleur": "#436CAE"},
    {"nom": "Bloc Droite", "x": 0.60, "y": -0.25, "sigma": 0.20, "poids": 30.0, "couleur": "#436CAE"},
    {"nom": "Bloc Périphérie", "x": 0.40, "y": 0.60, "sigma": 0.35, "poids": 15.0, "couleur": "#436CAE"},
]


def reinitialiser_modele_spatial():
    st.session_state.candidats_spatiaux = [c.copy() for c in CANDIDATS_SPATIAUX_DEFAUT]
    st.session_state.clusters_spatiaux = [cl.copy() for cl in CLUSTERS_SPATIAUX_DEFAUT]


# =============================================================================
# INITIALISATION DU SESSION STATE
# =============================================================================
if "population" not in st.session_state:
    st.session_state.population = None
if "liste_candidats" not in st.session_state:
    st.session_state.liste_candidats = list(CANDIDATS.values())
if "sondage_complet" not in st.session_state:
    st.session_state.sondage_complet = None
if "sondage_biaise" not in st.session_state:
    st.session_state.sondage_biaise = None
if "historique_resultats" not in st.session_state:
    st.session_state.historique_resultats = []
if "matrice_comparative" not in st.session_state:
    st.session_state.matrice_comparative = None
if "export_dir" not in st.session_state:
    st.session_state.export_dir = "./outputs/"
if "stv_rounds_details" not in st.session_state:
    st.session_state.stv_rounds_details = {}

# Session state Modèle Spatial 2D
if "candidats_spatiaux" not in st.session_state:
    st.session_state.candidats_spatiaux = [c.copy() for c in CANDIDATS_SPATIAUX_DEFAUT]
if "clusters_spatiaux" not in st.session_state:
    st.session_state.clusters_spatiaux = [cl.copy() for cl in CLUSTERS_SPATIAUX_DEFAUT]


# =============================================================================
# UTILITAIRES COULEURS
# =============================================================================
def hex_to_rgba(hex_color, alpha=0.15):
    """Convertit un code hexadécimal (#RRGGBB) en chaîne rgba(...) pour Plotly."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(67, 108, 175, {alpha})"


# =============================================================================
# DIALOGUES (MODALES DE MODIFICATION / AJOUT)
# =============================================================================
@st.dialog("Modifier le candidat")
def modal_modifier_candidat(idx):
    c = st.session_state.candidats_spatiaux[idx]
    nom = st.text_input("Nom :", value=c["nom"])
    col_x, col_y = st.columns(2)
    with col_x:
        x = st.number_input("Position X (Éco) :", min_value=-1.0, max_value=1.0, value=float(c["x"]), step=0.05)
    with col_y:
        y = st.number_input("Position Y (Soc) :", min_value=-1.0, max_value=1.0, value=float(c["y"]), step=0.05)
    couleur = st.color_picker("Couleur :", value=c["couleur"])

    if st.button("Enregistrer les modifications", type="primary", use_container_width=True):
        if not nom.strip():
            st.error("Le nom ne peut pas être vide.")
        else:
            st.session_state.candidats_spatiaux[idx] = {
                "nom": nom.strip(),
                "x": round(x, 2),
                "y": round(y, 2),
                "couleur": couleur
            }
            st.rerun()


@st.dialog("Ajouter un candidat")
def modal_ajouter_candidat():
    nom = st.text_input("Nom :", value="Nouveau")
    col_x, col_y = st.columns(2)
    with col_x:
        x = st.number_input("Position X (Éco) :", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
    with col_y:
        y = st.number_input("Position Y (Soc) :", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
    couleur = st.color_picker("Couleur :", value="#3366cc")

    if st.button("Ajouter", type="primary", use_container_width=True):
        noms_existants = [c["nom"] for c in st.session_state.candidats_spatiaux]
        if not nom.strip():
            st.error("Le nom ne peut pas être vide.")
        elif nom.strip() in noms_existants:
            st.error("Ce nom de candidat existe déjà.")
        else:
            st.session_state.candidats_spatiaux.append({
                "nom": nom.strip(),
                "x": round(x, 2),
                "y": round(y, 2),
                "couleur": couleur
            })
            st.rerun()


@st.dialog("Modifier le cluster")
def modal_modifier_cluster(idx):
    cl = st.session_state.clusters_spatiaux[idx]
    nom = st.text_input("Nom du cluster :", value=cl["nom"])
    col_x, col_y = st.columns(2)
    with col_x:
        x = st.number_input("Centre X :", min_value=-1.0, max_value=1.0, value=float(cl["x"]), step=0.05)
    with col_y:
        y = st.number_input("Centre Y :", min_value=-1.0, max_value=1.0, value=float(cl["y"]), step=0.05)
    sigma = st.slider("Dispersion (σ) :", 0.05, 0.60, float(cl["sigma"]), 0.01)
    
    col_p, col_c = st.columns(2)
    with col_p:
        poids = st.number_input("Poids (%) :", min_value=1.0, max_value=100.0, value=float(cl["poids"]), step=1.0)
    with col_c:
        couleur = st.color_picker("Couleur :", value=cl.get("couleur", "#436CAE"))

    if st.button("Enregistrer les modifications", type="primary", use_container_width=True):
        if not nom.strip():
            st.error("Le nom ne peut pas être vide.")
        else:
            st.session_state.clusters_spatiaux[idx] = {
                "nom": nom.strip(),
                "x": round(x, 2),
                "y": round(y, 2),
                "sigma": round(sigma, 2),
                "poids": round(poids, 1),
                "couleur": couleur
            }
            st.rerun()


@st.dialog("Ajouter un cluster")
def modal_ajouter_cluster():
    nom = st.text_input("Nom :", value="Nouveau Groupe")
    col_x, col_y = st.columns(2)
    with col_x:
        x = st.number_input("Centre X :", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
    with col_y:
        y = st.number_input("Centre Y :", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
    sigma = st.slider("Dispersion (σ) :", 0.05, 0.60, 0.25, 0.01)
    
    col_p, col_c = st.columns(2)
    with col_p:
        poids = st.number_input("Poids (%) :", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    with col_c:
        couleur = st.color_picker("Couleur :", value="#436CAE")

    if st.button("Ajouter", type="primary", use_container_width=True):
        if not nom.strip():
            st.error("Le nom ne peut pas être vide.")
        else:
            st.session_state.clusters_spatiaux.append({
                "nom": nom.strip(),
                "x": round(x, 2),
                "y": round(y, 2),
                "sigma": round(sigma, 2),
                "poids": round(poids, 1),
                "couleur": couleur
            })
            st.rerun()


# =============================================================================
# FONCTIONS UTILITAIRES DE SIMULATION
# =============================================================================
def calculer_metriques_injustice(pop, elu):
    """Calcule le regret individuel moyen et l'écart-type d'utilité."""
    utilites_elues = [e.utilities[elu] for e in pop]
    regrets = [max(e.utilities.values()) - e.utilities[elu] for e in pop]
    return float(np.mean(regrets)), float(np.std(utilites_elues))


def executer_scrutin_individuel(systeme_nom, pop, candidats, taux_strat, sondage, seuil_appr=50):
    """Exécute un seul scrutin et renvoie le vainqueur et les stats."""
    if systeme_nom == "Plurality":
        elu, details = vote_plurality(pop, candidats, taux_strat, sondage)
    elif systeme_nom == "Two-Round":
        elu, details, _ = vote_two_round(pop, candidats, taux_strat, sondage)
    elif systeme_nom == "STV":
        elu, details = vote_stv(pop, candidats, taux_strat, sondage)
    else:  # Approval
        elu, details = vote_approval(pop, candidats, taux_strat, sondage, seuil_defaut=seuil_appr)

    sw = calculer_social_welfare(pop, elu)
    regret_moyen, std_u = calculer_metriques_injustice(pop, elu)
    return elu, sw, regret_moyen, std_u, details


def lancer_simulation_complete(pop, candidats, seuil_appr=50):
    """
    Exécute les 5 cas demandés sur les 4 systèmes de vote en enregistrant
    les scores détaillés et métriques pour chaque scrutin.
    """
    s_complet = generer_sondage(pop, candidats, taille_echantillon=len(pop))
    taille_biaisee = max(15, int(len(pop) * 0.02))
    s_biaise = generer_sondage(pop, candidats, taille_echantillon=taille_biaisee)

    st.session_state.sondage_complet = s_complet
    st.session_state.sondage_biaise = s_biaise

    cand_opt, sw_opt, _ = calculer_optimum_social(pop, candidats)
    n_pop = len(pop)

    cas_definitions = [
        ("1. Sincère", 0.0, None),
        ("2. 100% Strat. (Sondage Complet)", 1.0, s_complet),
        ("3. 100% Strat. (Sondage Biaisé)", 1.0, s_biaise),
        ("4. 50% Strat. (Sondage Complet)", 0.5, s_complet),
        ("5. 50% Strat. (Sondage Biaisé)", 0.5, s_biaise)
    ]

    systemes_ordre = ["Plurality", "Two-Round", "STV", "Approval"]
    lignes = []
    details_par_systeme = {s: {} for s in systemes_ordre}
    stv_rounds_details = {}

    for nom_cas, taux, sondage in cas_definitions:
        elu_p, scores_p = vote_plurality(pop, candidats, taux, sondage)
        sw_p = calculer_social_welfare(pop, elu_p)
        details_par_systeme["Plurality"][nom_cas] = {c: (scores_p.get(c, 0) / n_pop) * 100 for c in candidats}

        elu_tr, t1_tr, _ = vote_two_round(pop, candidats, taux, sondage)
        sw_tr = calculer_social_welfare(pop, elu_tr)
        details_par_systeme["Two-Round"][nom_cas] = {c: (t1_tr["scores"].get(c, 0) / n_pop) * 100 for c in candidats}

        elu_stv, hist_stv = vote_stv(pop, candidats, taux, sondage)
        sw_stv = calculer_social_welfare(pop, elu_stv)
        t1_stv = hist_stv[0] if hist_stv else {}
        details_par_systeme["STV"][nom_cas] = {c: (t1_stv.get(c, 0) / n_pop) * 100 for c in candidats}
        stv_rounds_details[nom_cas] = [
            {c: (tour_dict.get(c, 0) / n_pop) * 100 for c in candidats}
            for tour_dict in hist_stv
        ]

        elu_app, scores_app = vote_approval(pop, candidats, taux, sondage, seuil_defaut=seuil_appr)
        sw_app = calculer_social_welfare(pop, elu_app)
        details_par_systeme["Approval"][nom_cas] = {c: (scores_app.get(c, 0) / n_pop) * 100 for c in candidats}

        resultats_cas = {
            "Plurality": (elu_p, sw_p),
            "Two-Round": (elu_tr, sw_tr),
            "STV": (elu_stv, sw_stv),
            "Approval": (elu_app, sw_app)
        }

        for sys_nom in systemes_ordre:
            elu, sw = resultats_cas[sys_nom]
            perte = sw_opt - sw
            pct_perte = (perte / sw_opt) * 100 if sw_opt > 0 else 0.0
            regret_moyen, std_u = calculer_metriques_injustice(pop, elu)

            lignes.append({
                "Cas": nom_cas,
                "Système": sys_nom,
                "Taux Stratégique": f"{int(taux * 100)}%",
                "Vainqueur": elu,
                "Bien-être Social Total": round(sw, 1),
                "Bien-être Social Moyen": round(sw / n_pop, 2),
                "Perte Absolue": round(perte, 1),
                "Perte (%)": round(pct_perte, 2),
                "Regret Moyen": round(regret_moyen, 2),
                "Inégalité (σ)": round(std_u, 2),
                "Optimal ?": "OUI" if elu == cand_opt else "NON"
            })

    df_resultats = pd.DataFrame(lignes)
    st.session_state.matrice_comparative = df_resultats
    st.session_state.stv_rounds_details = stv_rounds_details

    regret_opt, std_opt = calculer_metriques_injustice(pop, cand_opt)
    st.session_state.optimum_info = {
        "candidat": cand_opt,
        "sw_total": sw_opt,
        "sw_moyen": sw_opt / n_pop,
        "regret_moyen": regret_opt,
        "std_u": std_opt
    }
    st.session_state.details_par_systeme = details_par_systeme
    return df_resultats


# =============================================================================
# BARRE LATÉRALE
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller vers :",
    ["1. Population", "2. Simulation", "3. Résultats & Audit", "4. Paramètres & Export"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("État du Système")
if st.session_state.population is not None:
    st.sidebar.success(f"Population : {len(st.session_state.population)} agents")
else:
    st.sidebar.warning("Aucune population générée.")

if st.session_state.sondage_complet is not None:
    st.sidebar.info("Sondage complet disponible")
if st.session_state.sondage_biaise is not None:
    st.sidebar.info("Sondage biaisé disponible")


# =============================================================================
# PAGE 1 : POPULATION & CONFIGURATION
# =============================================================================
if page == "1. Population":
    st.title("Génération et Inspection de la Population")

    tab_spatial, tab_factions = st.tabs([
        "🧭 Modèle Spatial 2D",
        "📊 Modèle par Factions (Historique)"
    ])

    # =========================================================================
    # ONGLET 1 : MODÈLE SPATIAL 2D
    # =========================================================================
    with tab_spatial:
        st.markdown(
            "Configurez les candidats et les clusters d'électeurs sur l'échiquier politique 2D "
            "($X$ : Axe Économique $[-1, 1]$, $Y$ : Axe Sociétal $[-1, 1]$)."
        )

        col_spatial_cfg, col_spatial_vis = st.columns([1, 1], gap="medium")

        # ---------------------------------------------------------------------
        # COLONNE GAUCHE : PARAMÉTRAGE
        # ---------------------------------------------------------------------
        with col_spatial_cfg:
            st.subheader("1. Paramètres Généraux")
            n_electeurs_spat = st.slider(
                "Nombre d'électeurs (N) :",
                min_value=100, max_value=5000, value=1000, step=100,
                key="slider_n_spat"
            )
            seed_spat = st.number_input(
                "Graine aléatoire (Seed) :",
                value=42, step=1,
                key="seed_spat"
            )

            st.button(
                "🔄 Réinitialiser au Cas d'École par Défaut",
                on_click=reinitialiser_modele_spatial,
                use_container_width=True
            )

            st.markdown("---")

            # --- DÉROULANT : GESTION DES CANDIDATS ---
            with st.expander("👤 2. Gestion des Candidats", expanded=True):
                candidats = st.session_state.candidats_spatiaux

                for idx, c in enumerate(candidats):
                    col_info, col_btn_m, col_btn_d = st.columns([3, 1, 1])
                    with col_info:
                        st.markdown(
                            f"<span style='color:{c['couleur']}; font-weight:bold;'>● {c['nom']}</span> "
                            f"<code>({c['x']}, {c['y']})</code>",
                            unsafe_allow_html=True
                        )
                    with col_btn_m:
                        if st.button("✏️", key=f"btn_edit_c_{idx}", help="Modifier"):
                            modal_modifier_candidat(idx)
                    with col_btn_d:
                        if st.button("🗑️", key=f"btn_del_c_{idx}", help="Supprimer"):
                            if len(candidats) <= 2:
                                st.toast("Il faut au minimum 2 candidats !", icon="⚠️")
                            else:
                                st.session_state.candidats_spatiaux.pop(idx)
                                st.rerun()

                if st.button("➕ Ajouter un Candidat", use_container_width=True):
                    modal_ajouter_candidat()

            # --- DÉROULANT : GESTION DES CLUSTERS ---
            with st.expander("👥 3. Gestion des Clusters d'Électeurs", expanded=True):
                clusters = st.session_state.clusters_spatiaux
                total_poids = sum(cl["poids"] for cl in clusters)

                if abs(total_poids - 100.0) > 0.1:
                    st.caption(f"⚠️ Somme des poids : **{total_poids:.1f}%** (normalisée à 100% à la génération)")
                else:
                    st.caption("✅ Somme des poids : **100%**")

                for idx, cl in enumerate(clusters):
                    col_info_cl, col_btn_m_cl, col_btn_d_cl = st.columns([3, 1, 1])
                    with col_info_cl:
                        couleur_cl = cl.get("couleur", "#436CAE")
                        st.markdown(
                            f"<span style='color:{couleur_cl}; font-weight:bold;'>● {cl['nom']}</span> "
                            f"({cl['poids']}%) <code>({cl['x']}, {cl['y']}) σ={cl['sigma']}</code>",
                            unsafe_allow_html=True
                        )
                    with col_btn_m_cl:
                        if st.button("✏️", key=f"btn_edit_cl_{idx}", help="Modifier"):
                            modal_modifier_cluster(idx)
                    with col_btn_d_cl:
                        if st.button("🗑️", key=f"btn_del_cl_{idx}", help="Supprimer"):
                            if len(clusters) <= 1:
                                st.toast("Il faut au minimum 1 cluster !", icon="⚠️")
                            else:
                                st.session_state.clusters_spatiaux.pop(idx)
                                st.rerun()

                if st.button("➕ Ajouter un Cluster", use_container_width=True):
                    modal_ajouter_cluster()

            st.markdown("---")
            btn_gen_spat = st.button(
                "🚀 Générer la Population Spatiale",
                type="primary",
                use_container_width=True
            )
            if btn_gen_spat:
                st.info("Le générateur spatial 2D sera connecté à l'Étape 2 !")

        # ---------------------------------------------------------------------
        # COLONNE DROITE : VISUALISATION PLOTLY
        # ---------------------------------------------------------------------
        with col_spatial_vis:
            st.subheader("Visualisation du Modèle Spatial")

            # Cases à cocher (Options d'affichage)
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                vis_candidats = st.checkbox("Afficher Candidats", value=True, key="chk_cand")
            with col_v2:
                vis_clusters = st.checkbox("Afficher Clusters", value=True, key="chk_clust")
            with col_v3:
                vis_electeurs = st.checkbox("Afficher Électeurs", value=True, key="chk_elec")

            # Construction de la figure Plotly
            fig_spatial = go.Figure()

            # Lignes d'axes orthogonaux médians
            fig_spatial.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
            fig_spatial.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

            # 1. Traces pour les clusters (zones d'influence avec couleur propre)
            if vis_clusters:
                theta = np.linspace(0, 2 * np.pi, 60)
                for cl in clusters:
                    cx, cy, r = cl["x"], cl["y"], cl["sigma"]
                    cl_col = cl.get("couleur", "#436CAE")
                    x_circle = cx + r * np.cos(theta)
                    y_circle = cy + r * np.sin(theta)

                    fig_spatial.add_trace(go.Scatter(
                        x=x_circle,
                        y=y_circle,
                        mode="lines",
                        fill="toself",
                        fillcolor=hex_to_rgba(cl_col, alpha=0.15),
                        line=dict(color=hex_to_rgba(cl_col, alpha=0.55), width=1.5, dash="dot"),
                        name=f"σ {cl['nom']}",
                        hoverinfo="skip",
                        showlegend=False
                    ))

                    fig_spatial.add_trace(go.Scatter(
                        x=[cl["x"]],
                        y=[cl["y"]],
                        mode="text",
                        text=[f"<b>{cl['nom']}</b><br>({cl['poids']}%)"],
                        textposition="middle center",
                        textfont=dict(size=11, color=cl_col),
                        name=cl["nom"],
                        hoverinfo="skip",
                        showlegend=False
                    ))

            # 2. Traces pour les électeurs (si déjà générés)
            if vis_electeurs:
                if st.session_state.population is not None and getattr(st.session_state.population[0], "position", None) is not None:
                    xs_elec = [e.position[0] for e in st.session_state.population]
                    ys_elec = [e.position[1] for e in st.session_state.population]
                    fig_spatial.add_trace(go.Scatter(
                        x=xs_elec,
                        y=ys_elec,
                        mode="markers",
                        marker=dict(size=4, color="#a0aec0", opacity=0.4),
                        name="Électeurs",
                        hoverinfo="skip",
                        showlegend=False
                    ))

            # 3. Traces pour les candidats
            if vis_candidats:
                for c in candidats:
                    fig_spatial.add_trace(go.Scatter(
                        x=[c["x"]],
                        y=[c["y"]],
                        mode="markers+text",
                        marker=dict(size=18, color=c["couleur"], line=dict(width=2, color="black")),
                        text=[f"<b>{c['nom']}</b>"],
                        textposition="top center",
                        textfont=dict(size=12, color=c["couleur"]),
                        name=c["nom"],
                        hovertemplate=f"<b>{c['nom']}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
                        showlegend=False
                    ))

            fig_spatial.update_layout(
                xaxis=dict(
                    title="Axe Économique (Gauche ◄► Droite)",
                    range=[-1.05, 1.05],
                    zeroline=False,
                    gridcolor="rgba(200, 200, 200, 0.3)"
                ),
                yaxis=dict(
                    title="Axe Sociétal (Conservateur ◄► Progressiste)",
                    range=[-1.05, 1.05],
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1,
                    gridcolor="rgba(200, 200, 200, 0.3)"
                ),
                height=530,
                margin=dict(l=20, r=20, t=30, b=20),
                dragmode="pan"
            )

            st.plotly_chart(
                fig_spatial,
                use_container_width=True,
                key="spatial_plot"
            )

    # =========================================================================
    # ONGLET 2 : MODÈLE PAR FACTIONS (HISTORIQUE)
    # =========================================================================
    with tab_factions:
        st.markdown("Génération basée sur le modèle par factions prédéfinies ou lois de Dirichlet.")
        col_cfg, col_vis = st.columns([1, 2])

        with col_cfg:
            st.subheader("Paramètres")
            type_pop = st.selectbox(
                "Modèle de génération :",
                ["Scénario Fixe", "Scénario Aléatoire (Dirichlet)"],
                key="sel_type_pop_legacy"
            )
            n_electeurs = st.slider("Nombre d'électeurs (N) :", min_value=100, max_value=5000, value=1000, step=100, key="slider_n_legacy")
            seed = st.number_input("Graine aléatoire (Seed) :", value=42, step=1, key="seed_legacy")

            if st.button("Générer la Population (Factions)", type="primary", key="btn_gen_legacy"):
                fixer_aleatoire(seed)
                if type_pop == "Scénario Fixe":
                    pop = generer_population(n_electeurs)
                else:
                    pop, _, _ = generer_election_aleatoire(n_electeurs)

                st.session_state.population = pop
                st.session_state.liste_candidats = list(CANDIDATS.values())
                st.session_state.sondage_complet = None
                st.session_state.sondage_biaise = None
                st.session_state.matrice_comparative = None
                st.session_state.stv_rounds_details = {}
                st.success(f"Population de {n_electeurs} électeurs générée avec succès !")

        with col_vis:
            st.subheader("Distribution et Profils Idéologiques")
            if st.session_state.population is not None:
                pop = st.session_state.population
                candidats_legacy = st.session_state.liste_candidats

                factions = [e.faction for e in pop]
                counts = pd.Series(factions).value_counts(normalize=True) * 100

                fig, ax = plt.subplots(figsize=(7, 3))
                counts.plot(kind="bar", ax=ax, color="#2E86AB")
                ax.set_ylabel("% de la population")
                ax.set_title("Répartition des Factions")
                st.pyplot(fig)
                plt.close()

                utilites_moyennes = {
                    c: np.mean([e.utilities.get(c, 0) for e in pop]) for c in candidats_legacy
                }
                cand_opt, sw_opt, _ = calculer_optimum_social(pop, candidats_legacy)

                st.markdown(f"**Candidat Socialement Optimal :** `{cand_opt}` (SW Moyen : `{utilites_moyennes[cand_opt]:.1f}/100`)")

                fig2, ax2 = plt.subplots(figsize=(7, 3))
                ax2.bar(utilites_moyennes.keys(), utilites_moyennes.values(), color="#48CAE4")
                ax2.axhline(utilites_moyennes[cand_opt], color="red", linestyle="--", label=f"Optimum ({cand_opt})")
                ax2.set_ylabel("Utilité moyenne cardinale (/100)")
                ax2.set_title("Attractivité globale des candidats")
                ax2.legend()
                st.pyplot(fig2)
                plt.close()
            else:
                st.info("Veuillez générer une population dans le panneau de gauche.")


# =============================================================================
# PAGE 2 : SIMULATION & SONDAGE
# =============================================================================
elif page == "2. Simulation":
    st.title("Simulateur de Scrutins")

    if st.session_state.population is None:
        st.warning("Veuillez d'abord générer une population sur la page '1. Population'.")
    else:
        pop = st.session_state.population
        candidats = st.session_state.liste_candidats

        st.subheader("Lancement Global des 5 Scénarios")
        st.markdown("""
        Exécute simultanément les 4 scrutins sur les 5 configurations :
        *1. Sincère | 2. 100% Strat. Complet | 3. 100% Strat. Biaisé | 4. 50% Strat. Complet | 5. 50% Strat. Biaisé*
        """)
        if st.button("Lancer la Simulation Complète (Tous les cas)", type="primary"):
            with st.spinner("Exécution des simulations en cours..."):
                df_res = lancer_simulation_complete(pop, candidats)
            st.success("Simulation complète terminée ! Consultez l'onglet 'Résultats & Audit'.")
            st.dataframe(df_res.head(8), use_container_width=True)

        st.markdown("---")

        st.subheader("Simulation Ciblée / Unitaire")
        col_sond, col_scrutin = st.columns(2)

        with col_sond:
            st.markdown("**1. Génération de Sondage Préalable**")
            mode_sondage = st.radio("Type d'information électorale :", ["Complet (100% de la pop)", "Échantillon réduit (Biaisé)"])
            if st.button("Calculer le sondage"):
                taille = len(pop) if "Complet" in mode_sondage else max(15, int(len(pop) * 0.02))
                sond = generer_sondage(pop, candidats, taille_echantillon=taille)
                st.session_state.sondage_actif = sond
                st.write("Résultats du sondage (Intentions 1er choix) :")
                st.json(sond)

        with col_scrutin:
            st.markdown("**2. Paramètres du Scrutin Spécifique**")
            sys_choisi = st.selectbox("Système de vote :", ["Plurality", "Two-Round", "STV", "Approval"])
            taux_strat = st.slider("Proportion de votants stratégiques :", 0.0, 1.0, 0.5, 0.05)
            seuil_app = st.slider("Seuil Approval Voting (sincère) :", 10, 90, 50, 5)

            if st.button("Lancer ce Scrutin"):
                sondage_a_utiliser = st.session_state.get("sondage_actif", None)
                if taux_strat > 0 and sondage_a_utiliser is None:
                    st.error("Pour un vote stratégique, calculez d'abord un sondage !")
                else:
                    elu, sw, regret, std_u, details = executer_scrutin_individuel(
                        sys_choisi, pop, candidats, taux_strat, sondage_a_utiliser, seuil_app
                    )
                    cand_opt, sw_opt, _ = calculer_optimum_social(pop, candidats)
                    loss = ((sw_opt - sw) / sw_opt) * 100

                    st.markdown(f"### Résultat : Élu **{elu}**")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Social Welfare", f"{sw:.1f}", delta=f"-{loss:.2f}% de perte")
                    col_m2.metric("Regret Moyen", f"{regret:.2f} pts")
                    col_m3.metric("Inégalité (σ)", f"{std_u:.2f}")
                    st.write("Détails d'exécution :", details)


# =============================================================================
# PAGE 3 : RÉSULTATS & AUDIT
# =============================================================================
elif page == "3. Résultats & Audit":
    st.title("Audit du Bien-être Social et Impact du Vote Stratégique")

    if st.session_state.matrice_comparative is None:
        st.info("Aucun résultat complet à afficher. Cliquez sur 'Lancer la Simulation Complète' sur la page Simulation.")
    else:
        df = st.session_state.matrice_comparative
        opt = st.session_state.optimum_info
        details_sys = st.session_state.get("details_par_systeme", {})
        stv_rounds = st.session_state.get("stv_rounds_details", {})

        ORDRE_SYSTEMES = ["Plurality", "Two-Round", "STV", "Approval"]
        ORDRE_CAS = [
            "1. Sincère",
            "2. 100% Strat. (Sondage Complet)",
            "3. 100% Strat. (Sondage Biaisé)",
            "4. 50% Strat. (Sondage Complet)",
            "5. 50% Strat. (Sondage Biaisé)"
        ]

        st.markdown(
            f"**Référence théorique (Optimum Social) :** Vainqueur **`{opt['candidat']}`** | "
            f"SW Moyen : **`{opt['sw_moyen']:.2f} / 100`** | "
            f"Regret Moyen : **`{opt['regret_moyen']:.2f}`** | "
            f"Dispersion $\\sigma$ : **`{opt['std_u']:.2f}`**"
        )

        tab_global, tab_plurality, tab_two_round, tab_stv, tab_approval = st.tabs([
            "📊 Vue Globale",
            "1. Plurality",
            "2. Two-Round",
            "3. STV",
            "4. Approval"
        ])

        # -------------------------------------------------------------
        # ONGLET 1 : VUE GLOBALE
        # -------------------------------------------------------------
        with tab_global:
            st.subheader("Matrices Comparatives (4 Systèmes × 5 Scénarios)")

            pivot_vainqueur = df.pivot(index="Cas", columns="Système", values="Vainqueur").reindex(
                index=ORDRE_CAS, columns=ORDRE_SYSTEMES
            )
            pivot_loss = df.pivot(index="Cas", columns="Système", values="Perte (%)").reindex(
                index=ORDRE_CAS, columns=ORDRE_SYSTEMES
            ).astype(float)
            pivot_regret = df.pivot(index="Cas", columns="Système", values="Regret Moyen").reindex(
                index=ORDRE_CAS, columns=ORDRE_SYSTEMES
            ).astype(float)
            pivot_sw_moyen = df.pivot(index="Cas", columns="Système", values="Bien-être Social Moyen").reindex(
                index=ORDRE_CAS, columns=ORDRE_SYSTEMES
            )

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Candidat Élu**")
                st.dataframe(pivot_vainqueur, use_container_width=True)
            with col_t2:
                st.markdown("**Perte de Bien-être Social (%)**")
                max_loss = max(1.0, float(pivot_loss.max().max()))
                st.dataframe(
                    pivot_loss.style.format("{:.2f}%").background_gradient(
                        cmap="Reds", vmin=0.0, vmax=max_loss
                    ),
                    use_container_width=True
                )

            st.markdown("---")
            st.subheader("Indicateurs d'Injustice et d'Inégalité")
            col_reg, col_gini = st.columns(2)
            with col_reg:
                st.markdown("**Regret Individuel Moyen (pts d'utilité perdus)**")
                st.dataframe(
                    pivot_regret.style.format("{:.2f}").background_gradient(cmap="Oranges"),
                    use_container_width=True
                )
            with col_gini:
                st.markdown("**Inégalité / Frustration ($sigma$ des utilités)**")
                pivot_std = df.pivot(index="Cas", columns="Système", values="Inégalité (σ)").reindex(
                    index=ORDRE_CAS, columns=ORDRE_SYSTEMES
                ).astype(float)
                st.dataframe(
                    pivot_std.style.format("{:.2f}").background_gradient(cmap="Blues"),
                    use_container_width=True
                )

            st.markdown("---")
            st.subheader("Comparaison du Bien-être Social Moyen par Habitant")

            fig, ax = plt.subplots(figsize=(10, 4.5))
            pivot_sw_plot = pivot_sw_moyen.T.reindex(ORDRE_SYSTEMES)
            pivot_sw_plot.plot(kind="bar", ax=ax)

            ax.axhline(opt["sw_moyen"], color="black", linestyle="--", linewidth=1.5, label=f"Optimum ({opt['candidat']})")
            ax.set_ylabel("Bien-être Social Moyen (/100)")
            ax.set_title("Bien-être Social moyen par électeur selon le mode de scrutin")
            ax.set_xticklabels(ORDRE_SYSTEMES, rotation=0)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            with st.expander("Voir les données brutes"):
                st.dataframe(df, use_container_width=True)

        # -------------------------------------------------------------
        # ONGLETS INDIVIDUELS
        # -------------------------------------------------------------
        onglets_systemes = [
            (tab_plurality, "Plurality", "Voix obtenues (%)"),
            (tab_two_round, "Two-Round", "Voix au 1er Tour (%)"),
            (tab_stv, "STV", "Voix de 1ère Préférence (%)"),
            (tab_approval, "Approval", "Taux d'Approbation (%)")
        ]

        for tab, sys_nom, label_y in onglets_systemes:
            with tab:
                st.subheader(f"Analyse détaillée : {sys_nom}")

                df_sys = df[df["Système"] == sys_nom].set_index("Cas").reindex(ORDRE_CAS)

                st.markdown("**Synthèse des Scénarios**")
                st.dataframe(
                    df_sys[["Vainqueur", "Bien-être Social Moyen", "Perte (%)", "Regret Moyen", "Inégalité (σ)", "Optimal ?"]],
                    use_container_width=True
                )

                st.markdown("---")

                st.markdown(f"**Comparaison par Candidat sous chaque scénario : {label_y}**")
                if sys_nom in details_sys and details_sys[sys_nom]:
                    df_details = pd.DataFrame(details_sys[sys_nom]).reindex(columns=ORDRE_CAS)

                    fig_sys, ax_sys = plt.subplots(figsize=(10, 4.5))
                    df_details.plot(kind="bar", ax=ax_sys)
                    ax_sys.set_ylabel(label_y)
                    ax_sys.set_title(f"Répartition par candidat selon le scénario ({sys_nom})")
                    ax_sys.set_xticklabels(df_details.index, rotation=0)
                    ax_sys.legend(title="Scénario", bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.tight_layout()
                    st.pyplot(fig_sys)
                    plt.close()

                if sys_nom == "STV" and stv_rounds:
                    st.markdown("---")
                    st.subheader("Cascade d'élimination et Reports de voix (STV)")
                    scen_choisi = st.selectbox(
                        "Choisir un scénario pour voir l'historique des transferts :",
                        ORDRE_CAS,
                        key="select_scen_stv"
                    )

                    tours_data = stv_rounds.get(scen_choisi, [])
                    if tours_data:
                        df_rounds = pd.DataFrame(tours_data)
                        df_rounds.index = [f"Tour {i+1}" for i in range(len(df_rounds))]

                        col_stv_g, col_stv_t = st.columns([2, 1])

                        with col_stv_g:
                            fig_stv, ax_stv = plt.subplots(figsize=(8, 4))
                            df_rounds.plot(kind="bar", stacked=True, ax=ax_stv, colormap="tab10")
                            ax_stv.set_ylabel("% des voix actives")
                            ax_stv.set_title(f"Éliminations et reports successifs ({scen_choisi})")
                            ax_stv.set_xticklabels(df_rounds.index, rotation=0)
                            ax_stv.axhline(50.0, color="black", linestyle="--", linewidth=1, label="Majorité absolue (50%)")
                            ax_stv.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                            plt.tight_layout()
                            st.pyplot(fig_stv)
                            plt.close()

                        with col_stv_t:
                            st.markdown("**Voix par tour (%)**")
                            st.dataframe(df_rounds.round(1), use_container_width=True)


# =============================================================================
# PAGE 4 : PARAMÈTRES & EXPORT
# =============================================================================
elif page == "4. Paramètres & Export":
    st.title("Paramètres et Sauvegarde des Données")

    st.subheader("Répertoire d'exportation")
    export_path = st.text_input("Chemin du dossier de sortie :", value=st.session_state.export_dir)
    st.session_state.export_dir = export_path

    if not os.path.exists(export_path):
        if st.button("Créer le dossier d'export"):
            os.makedirs(export_path, exist_ok=True)
            st.success(f"Dossier '{export_path}' créé avec succès !")

    st.markdown("---")
    st.subheader("Export des Résultats")

    if st.session_state.matrice_comparative is not None:
        df = st.session_state.matrice_comparative
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        nom_fichier = f"simulation_audit_{horodatage}.csv"

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("Enregistrer sur le disque (CSV)"):
                os.makedirs(export_path, exist_ok=True)
                full_path = os.path.join(export_path, nom_fichier)
                df.to_csv(full_path, index=False)
                st.success(f"Fichier sauvegardé : `{full_path}`")

        with col_exp2:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Télécharger directement le CSV",
                data=csv_data,
                file_name=nom_fichier,
                mime="text/csv"
            )
    else:
        st.info("Aucun résultat de simulation complète disponible pour l'export.")