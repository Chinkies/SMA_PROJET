"""
app.py - Interface Streamlit pour la simulation des systèmes de vote
et l'analyse du vote stratégique (Projet 4).
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# =============================================================================
# FONCTIONS UTILITAIRES DE SIMULATION
# =============================================================================
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
    return elu, sw, details


def lancer_simulation_complete(pop, candidats, seuil_appr=50):
    """
    Exécute les 5 cas demandés sur les 4 systèmes de vote en enregistrant
    les scores détaillés pour chaque scrutin.
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

    for nom_cas, taux, sondage in cas_definitions:
        # 1. Plurality
        elu_p, scores_p = vote_plurality(pop, candidats, taux, sondage)
        sw_p = calculer_social_welfare(pop, elu_p)
        details_par_systeme["Plurality"][nom_cas] = {c: (scores_p.get(c, 0) / n_pop) * 100 for c in candidats}

        # 2. Two-Round
        elu_tr, t1_tr, _ = vote_two_round(pop, candidats, taux, sondage)
        sw_tr = calculer_social_welfare(pop, elu_tr)
        details_par_systeme["Two-Round"][nom_cas] = {c: (t1_tr["scores"].get(c, 0) / n_pop) * 100 for c in candidats}

        # 3. STV
        elu_stv, hist_stv = vote_stv(pop, candidats, taux, sondage)
        sw_stv = calculer_social_welfare(pop, elu_stv)
        # On extrait le 1er tour de STV pour observer l'effet du report/stratégie
        t1_stv = hist_stv[0] if hist_stv else {}
        details_par_systeme["STV"][nom_cas] = {c: (t1_stv.get(c, 0) / n_pop) * 100 for c in candidats}

        # 4. Approval
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
            lignes.append({
                "Cas": nom_cas,
                "Système": sys_nom,
                "Taux Stratégique": f"{int(taux * 100)}%",
                "Vainqueur": elu,
                "Bien-être Social Total": round(sw, 1),
                "Bien-être Social Moyen": round(sw / n_pop, 2),
                "Perte Absolue": round(perte, 1),
                "Perte (%)": round(pct_perte, 2),
                "Optimal ?": "OUI" if elu == cand_opt else "NON"
            })

    df_resultats = pd.DataFrame(lignes)
    st.session_state.matrice_comparative = df_resultats
    st.session_state.optimum_info = {
        "candidat": cand_opt,
        "sw_total": sw_opt,
        "sw_moyen": sw_opt / n_pop
    }
    st.session_state.details_par_systeme = details_par_systeme
    return df_resultats


# =============================================================================
# BARRE LATÉRALE : NAVIGATION ET STATUT
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
# PAGE 1 : POPULATION & VISUALISATION
# =============================================================================
if page == "1. Population":
    st.title("Génération et Inspection de la Population")

    col_cfg, col_vis = st.columns([1, 2])

    with col_cfg:
        st.subheader("Paramètres")
        type_pop = st.selectbox("Modèle de génération :", ["Scénario Fixe", "Scénario Aléatoire (Dirichlet)"])
        n_electeurs = st.slider("Nombre d'électeurs (N) :", min_value=100, max_value=5000, value=1000, step=100)
        seed = st.number_input("Graine aléatoire (Seed) :", value=42, step=1)

        if st.button("Générer la Population", type="primary"):
            fixer_aleatoire(seed)
            if type_pop == "Scénario Fixe":
                pop = generer_population(n_electeurs)
            else:
                pop, _, _ = generer_election_aleatoire(n_electeurs)

            st.session_state.population = pop
            # Invalidation des anciens résultats
            st.session_state.sondage_complet = None
            st.session_state.sondage_biaise = None
            st.session_state.matrice_comparative = None
            st.success(f"Population de {n_electeurs} électeurs générée avec succès !")

    with col_vis:
        st.subheader("Distribution et Profils Idéologiques")
        if st.session_state.population is not None:
            pop = st.session_state.population
            candidats = st.session_state.liste_candidats

            # 1. Barplot des Factions
            factions = [e.faction for e in pop]
            counts = pd.Series(factions).value_counts(normalize=True) * 100

            fig, ax = plt.subplots(figsize=(7, 3))
            counts.plot(kind="bar", ax=ax, color="#2E86AB")
            ax.set_ylabel("% de la population")
            ax.set_title("Répartition des Factions")
            st.pyplot(fig)
            plt.close()

            # 2. Utilités moyennes par candidat
            utilites_moyennes = {
                c: np.mean([e.utilities[c] for e in pop]) for c in candidats
            }
            cand_opt, sw_opt, _ = calculer_optimum_social(pop, candidats)

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

        # -------------------------------------------------------------
        # Bouton MAJEUR : Simulation Complète
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Simulation Unitaire / Test ciblé
        # -------------------------------------------------------------
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
                    elu, sw, details = executer_scrutin_individuel(
                        sys_choisi, pop, candidats, taux_strat, sondage_a_utiliser, seuil_app
                    )
                    cand_opt, sw_opt, _ = calculer_optimum_social(pop, candidats)
                    loss = ((sw_opt - sw) / sw_opt) * 100

                    st.markdown(f"### Résultat : Élu **{elu}**")
                    st.metric(label="Social Welfare", value=f"{sw:.1f}", delta=f"-{loss:.2f}% de perte")
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

        # Ordre strict exigé
        ORDRE_SYSTEMES = ["Plurality", "Two-Round", "STV", "Approval"]
        ORDRE_CAS = [
            "1. Sincère",
            "2. 100% Strat. (Sondage Complet)",
            "3. 100% Strat. (Sondage Biaisé)",
            "4. 50% Strat. (Sondage Complet)",
            "5. 50% Strat. (Sondage Biaisé)"
        ]

        # Bandeau de référence
        st.markdown(
            f"**Référence théorique (Optimum Social) :** Vainqueur **`{opt['candidat']}`** "
            f"avec un Bien-être Social moyen de **`{opt['sw_moyen']:.2f} / 100`** "
            f"*(Total : `{opt['sw_total']:.1f}`)*"
        )

        # Création des onglets
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
            )
            pivot_sw_moyen = df.pivot(index="Cas", columns="Système", values="Bien-être Social Moyen").reindex(
                index=ORDRE_CAS, columns=ORDRE_SYSTEMES
            )

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Candidat Élu**")
                st.dataframe(pivot_vainqueur, use_container_width=True)
            with col_t2:
                st.markdown("**Perte de Bien-être Social (%)**")
                st.dataframe(pivot_loss.style.format("{:.2f}%").background_gradient(cmap="Reds"), use_container_width=True)

            st.markdown("---")
            st.subheader("Comparaison du Bien-être Social Moyen par Habitant")

            # Graphique réindexé selon ORDRE_SYSTEMES
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
        # ONGLETS INDIVIDUELS PAR SYSTÈME
        # -------------------------------------------------------------
        onglets_systemes = [
            (tab_plurality, "Plurality", "Voix obtenues (%)"),
            (tab_two_round, "Two-Round", "Voix au 1er Tour (%)"),
            (tab_stv, "STV", "Voix de 1ère Préférence (Tour 1) (%)"),
            (tab_approval, "Approval", "Taux d'Approbation (%)")
        ]

        for tab, sys_nom, label_y in onglets_systemes:
            with tab:
                st.subheader(f"Analyse détaillée : {sys_nom}")

                # Filtre des données du système
                df_sys = df[df["Système"] == sys_nom].set_index("Cas").reindex(ORDRE_CAS)

                col_res, col_chart = st.columns([1, 2])

                with col_res:
                    st.markdown("**Synthèse des Scénarios**")
                    st.dataframe(
                        df_sys[["Vainqueur", "Bien-être Social Moyen", "Perte (%)", "Optimal ?"]],
                        use_container_width=True
                    )

                with col_chart:
                    st.markdown(f"**Évolution des scores des candidats : {label_y}**")
                    if sys_nom in details_sys and details_sys[sys_nom]:
                        df_details = pd.DataFrame(details_sys[sys_nom]).T.reindex(ORDRE_CAS)

                        fig_sys, ax_sys = plt.subplots(figsize=(8, 4))
                        df_details.plot(kind="bar", ax=ax_sys)
                        ax_sys.set_ylabel(label_y)
                        ax_sys.set_title(f"Répartition des choix par candidat ({sys_nom})")
                        ax_sys.set_xticklabels(ORDRE_CAS, rotation=25, ha="right")
                        ax_sys.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                        plt.tight_layout()
                        st.pyplot(fig_sys)
                        plt.close()


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