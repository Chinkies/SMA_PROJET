"""
benchmark.py - Suite complète de benchmarks Monte-Carlo :
1. Courbes d'utilité (% de l'Optimum) vs % de votants stratégiques (Sondage Complet)
2. Courbes d'utilité (% de l'Optimum) vs % de votants stratégiques (Sondage Biaisé 200/10000)
3. Diagramme en bâtons : Échantillon de 100 agents Statiques vs 100 agents IA (Ollama) sur Pop=10000
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from population import (
    generer_election_spatiale_aleatoire,
    generer_sondage,
    fixer_aleatoire,
)
from voting_systems import (
    auditer_election,
    calculer_social_welfare,
    vote_plurality,
    vote_two_round,
    vote_stv,
    vote_approval,
)
from llm_agents import preparer_prompts_llm, executer_mode_llm, OLLAMA_MODEL

# Dossier de sauvegarde automatique des figures pour le rapport PDF
OUTPUT_DIR = "./outputs_benchmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# GRAPHIQUES 1 & 2 : COURBES DE SATISFACTION SELON LE % DE VOTE STRATÉGIQUE
# =============================================================================
def run_benchmark_courbes(nb_simulations=40, pop_size=10000, taille_sondage=None, titre="", nom_fichier="courbe.png"):
    """
    À chaque simulation :
      - Génère une NOUVELLE population aléatoire spatiale (nouveaux clusters et candidats)
      - Génère un NOUVEAU sondage (complet ou biaisé sur 200 personnes)
      - Évalue les 4 modes de scrutin pour 11 paliers de vote stratégique (0% à 100%)
    """
    print(f"\n==================================================================")
    print(f" DÉMARRAGE : {titre}")
    print(f" Paramètres : {nb_simulations} simulations | Pop={pop_size} | Sondage={taille_sondage or pop_size}")
    print(f"==================================================================")

    taux_values = np.linspace(0.0, 1.0, 11)  # 0%, 10%, ..., 100%
    systemes = ["Plurality", "Two-Round", "STV", "Approval"]

    # Stockage des scores de satisfaction (% par rapport à l'Optimum Social)
    resultats_sat = {s: np.zeros(11) for s in systemes}

    for sim in range(nb_simulations):
        if (sim + 1) % 5 == 0 or sim == 0:
            print(f"  -> Simulation aléatoire {sim + 1}/{nb_simulations}...")

        # Nouvelle graine aléatoire à chaque itération
        fixer_aleatoire(1000 + sim * 37 + random.randint(1, 999))

        # 1. Génération d'une nouvelle population spatiale aléatoire
        pop, cands_info, _ = generer_election_spatiale_aleatoire(pop_size)
        candidats_noms = [c["nom"] for c in cands_info]

        # 2. Génération d'un nouveau sondage (aléatoire / biaisé si taille_sondage < pop_size)
        sondage = generer_sondage(pop, candidats_noms, taille_echantillon=taille_sondage)

        # 3. Évaluation sur les 11 paliers de stratèges
        for i, taux in enumerate(taux_values):
            rapport = auditer_election(pop, candidats_noms, taux_strategique=taux, sondage=sondage)
            for sys_nom in systemes:
                # On utilise la satisfaction brute (% de l'Optimum Social)
                resultats_sat[sys_nom][i] += rapport["resultats"][sys_nom]["satisfaction_brute"]

    # Moyenne sur l'ensemble des simulations
    for sys_nom in systemes:
        resultats_sat[sys_nom] /= nb_simulations

    # --- TRACÉ DU GRAPHIQUE ---
    plt.figure(figsize=(10, 6))
    couleurs = {
        "Plurality": "#ff7f0e",   # Orange
        "Two-Round": "#d62728",   # Rouge
        "STV": "#2ca02c",         # Vert
        "Approval": "#1f77b4"     # Bleu
    }
    marqueurs = {"Plurality": "o", "Two-Round": "s", "STV": "^", "Approval": "D"}

    for sys_nom in systemes:
        plt.plot(
            taux_values * 100,
            resultats_sat[sys_nom],
            marker=marqueurs[sys_nom],
            label=sys_nom,
            color=couleurs[sys_nom],
            linewidth=2.5,
            markersize=7
        )

    plt.title(titre, fontsize=14, fontweight="bold")
    plt.xlabel("Pourcentage de votants stratégiques (%)", fontsize=12)
    plt.ylabel("Score d'Utilité Sociale (% de l'Optimum)", fontsize=12)
    plt.xticks(np.arange(0, 110, 10))

    min_y = min(min(v) for v in resultats_sat.values())
    plt.ylim(bottom=max(0.0, min_y - 2.0), top=100.5)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()

    chemin_fig = os.path.join(OUTPUT_DIR, nom_fichier)
    plt.savefig(chemin_fig, dpi=300)
    print(f"  [✅ Graphique sauvegardé dans : {chemin_fig}]")
    plt.show()


# =============================================================================
# GRAPHIQUE 3 : COMPARAISON ÉCHANTILLON STATIQUE VS OLLAMA (BAR CHART)
# =============================================================================
def evaluer_echantillon_ollama_sur_scrutin(pop, candidats, indices_strat, sondage, systeme):
    """
    Fait voter les 100 agents de `indices_strat` via l'agent IA (Ollama) et compte les bulletins
    avec le VRAI moteur électoral du scrutin correspondant.
    """
    echantillon_electeurs = [pop[idx] for idx in indices_strat]
    groupes, prompts = preparer_prompts_llm(echantillon_electeurs, sondage, systeme)
    choix_par_profil = executer_mode_llm(
        groupes, prompts, candidats, systeme_vote=systeme, sondage_actuel=sondage
    )

    votes_custom = {}
    for idx in indices_strat:
        electeur = pop[idx]
        profil_cle = tuple(electeur.get_classement())
        vote_llm = choix_par_profil.get(profil_cle)
        if vote_llm is not None:
            votes_custom[idx] = vote_llm

    if systeme == "Plurality":
        elu, _ = vote_plurality(pop, candidats, sondage=sondage, indices_strat=indices_strat, votes_custom=votes_custom)
    elif systeme == "Two-Round":
        elu, _, _ = vote_two_round(pop, candidats, sondage=sondage, indices_strat=indices_strat, votes_custom=votes_custom)
    elif systeme == "STV":
        elu, _ = vote_stv(pop, candidats, sondage=sondage, indices_strat=indices_strat, votes_custom=votes_custom)
    else:  # Approval
        elu, _ = vote_approval(pop, candidats, sondage=sondage, indices_strat=indices_strat, votes_custom=votes_custom)

    return elu


def run_benchmark_statique_vs_ollama(nb_simulations=10, pop_size=10000, nb_agents_strat=100, taille_sondage=500):
    """
    Compare sur la même population aléatoire (N=10000) et le même échantillon (100 agents) :
      - La décision stratégique d'agents Statiques (heuristiques programmées)
      - La décision stratégique d'agents IA (LLM local via Ollama)
    Mesure à la fois :
      1. Le Bien-être Social Global (% de l'Optimum)
      2. L'Utilité Individuelle Moyenne obtenue par l'échantillon des 100 agents stratégiques !
    """
    print(f"\n==================================================================")
    print(f" DÉMARRAGE : Comparaison Agents Statiques vs Agents IA (Ollama : {OLLAMA_MODEL})")
    print(f" Paramètres : {nb_simulations} simulations | Pop={pop_size} | Échantillon={nb_agents_strat} | Sondage={taille_sondage}")
    print(f"==================================================================")

    systemes = ["Plurality", "Two-Round", "STV", "Approval"]
    taux_strat = nb_agents_strat / float(pop_size)

    # 1. Satisfaction de la population globale (% de l'Optimum)
    sat_globale_stat = {s: 0.0 for s in systemes}
    sat_globale_ia = {s: 0.0 for s in systemes}

    # 2. Utilité moyenne obtenue par l'échantillon des 100 agents stratégiques eux-mêmes (/100)
    util_echantillon_stat = {s: 0.0 for s in systemes}
    util_echantillon_ia = {s: 0.0 for s in systemes}

    for sim in range(nb_simulations):
        print(f"\n--- Simulation {sim + 1}/{nb_simulations} ---")
        fixer_aleatoire(5000 + sim * 53 + random.randint(1, 999))

        # Génération d'une nouvelle population aléatoire et d'un nouveau sondage biaisé
        pop, cands_info, _ = generer_election_spatiale_aleatoire(pop_size)
        candidats = [c["nom"] for c in cands_info]
        sondage = generer_sondage(pop, candidats, taille_echantillon=taille_sondage)

        # Tirage de l'échantillon EXACT de 100 électeurs stratégiques
        indices_strat = set(random.sample(range(pop_size), nb_agents_strat))

        # Optimum social de cette population
        total_utilites = {c: calculer_social_welfare(pop, c) for c in candidats}
        sw_opt = max(total_utilites.values())

        # Audit Statique sur ce même échantillon de 100 agents
        rapport_stat = auditer_election(pop, candidats, taux_strategique=taux_strat, sondage=sondage, indices_strat=indices_strat)

        for sys_nom in systemes:
            # --- RÉSULTAT STATIQUE ---
            elu_stat = rapport_stat["resultats"][sys_nom]["vainqueur"]
            sat_globale_stat[sys_nom] += rapport_stat["resultats"][sys_nom]["satisfaction_brute"]
            util_echantillon_stat[sys_nom] += np.mean([pop[i].utilities[elu_stat] for i in indices_strat])

            # --- RÉSULTAT IA (OLLAMA) ---
            print(f"  [{sys_nom}] Consultation d'Ollama ({OLLAMA_MODEL}) pour l'échantillon de {nb_agents_strat} agents...")
            elu_ia = evaluer_echantillon_ollama_sur_scrutin(pop, candidats, indices_strat, sondage, sys_nom)
            sw_ia = total_utilites[elu_ia]
            sat_globale_ia[sys_nom] += (sw_ia / sw_opt) * 100.0
            util_echantillon_ia[sys_nom] += np.mean([pop[i].utilities[elu_ia] for i in indices_strat])

    # Moyennes
    vals_stat_glob = [sat_globale_stat[s] / nb_simulations for s in systemes]
    vals_ia_glob = [sat_globale_ia[s] / nb_simulations for s in systemes]
    vals_stat_ech = [util_echantillon_stat[s] / nb_simulations for s in systemes]
    vals_ia_ech = [util_echantillon_ia[s] / nb_simulations for s in systemes]

    # --- TRACÉ DU GRAPHIQUE EN BÂTONS (DOUBLE VUE : GLOBALE & ÉCHANTILLON) ---
    x = np.arange(len(systemes))
    width = 0.35
    labels_x = ["1T (Plurality)", "2T (Two-Round)", "STV (Ranked)", "APR (Approval)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sous-graphe 1 : Utilité Sociale Globale (% de l'Optimum)
    rects1 = ax1.bar(x - width / 2, vals_stat_glob, width, label="Agents Statiques (SIMPL)", color="#ff7f0e", edgecolor="black")
    rects2 = ax1.bar(x + width / 2, vals_ia_glob, width, label=f"Agents IA ({OLLAMA_MODEL})", color="#2ca02c", edgecolor="black")
    ax1.set_ylabel("Score d'Utilité Sociale (% de l'Optimum)", fontsize=11)
    ax1.set_title(f"Impact sur le Bien-être Global\n(Pop={pop_size}, Échantillon={nb_agents_strat}, Sondage={taille_sondage})", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_x)
    min_g = min(min(vals_stat_glob), min(vals_ia_glob))
    ax1.set_ylim(bottom=max(0.0, min_g - 3.0), top=100.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend()
    ax1.bar_label(rects1, fmt="%.1f%%", padding=3, fontsize=9)
    ax1.bar_label(rects2, fmt="%.1f%%", padding=3, fontsize=9)

    # Sous-graphe 2 : Satisfaction moyenne des 100 agents de l'échantillon stratégique (/100)
    rects3 = ax2.bar(x - width / 2, vals_stat_ech, width, label="Agents Statiques (SIMPL)", color="#ff7f0e", edgecolor="black")
    rects4 = ax2.bar(x + width / 2, vals_ia_ech, width, label=f"Agents IA ({OLLAMA_MODEL})", color="#2ca02c", edgecolor="black")
    ax2.set_ylabel("Utilité moyenne de l'échantillon stratégique (/100)", fontsize=11)
    ax2.set_title(f"Gain direct pour l'échantillon de {nb_agents_strat} stratèges\n(Qualité de réflexion Statique vs IA)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_x)
    min_e = min(min(vals_stat_ech), min(vals_ia_ech))
    ax2.set_ylim(bottom=max(0.0, min_e - 5.0), top=min(100.0, max(max(vals_stat_ech), max(vals_ia_ech)) + 6.0))
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.legend()
    ax2.bar_label(rects3, fmt="%.1f", padding=3, fontsize=9)
    ax2.bar_label(rects4, fmt="%.1f", padding=3, fontsize=9)

    plt.tight_layout()
    chemin_fig = os.path.join(OUTPUT_DIR, "3_comparaison_statique_vs_ollama.png")
    plt.savefig(chemin_fig, dpi=300)
    print(f"  [✅ Graphique sauvegardé dans : {chemin_fig}]")
    plt.show()


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Nombre de simulations par point pour lisser les courbes (ex: 30 à 50)
    NB_SIMULATIONS_COURBES = 30
    # Nombre de simulations pour le comparatif Ollama vs Statique (ex: 5 à 10)
    NB_SIMULATIONS_OLLAMA = 5

    """
    # 1. Courbe avec Sondage Complet (100% de la population = 10 000)
    run_benchmark_courbes(
        nb_simulations=NB_SIMULATIONS_COURBES,
        pop_size=10000,
        taille_sondage=10000,
        titre="Score d'Utilité Sociale selon le % de Votants Stratégiques\n(Sondage Complet : 10 000 / 10 000)",
        nom_fichier="1_courbes_sondage_complet.png"
    )

    # 2. Courbe avec Sondage Biaisé (Échantillon restreint = 200 / 10 000)
    run_benchmark_courbes(
        nb_simulations=NB_SIMULATIONS_COURBES,
        pop_size=10000,
        taille_sondage=200,
        titre="Score d'Utilité Sociale selon le % de Votants Stratégiques\n(Sondage Biaisé : 200 / 10 000 électeurs)",
        nom_fichier="2_courbes_sondage_biaise.png"
    )
    """
    # 3. Bar Chart : Échantillon de 100 agents Statiques vs 100 agents IA (Ollama) sur Pop=10 000
    run_benchmark_statique_vs_ollama(
        nb_simulations=NB_SIMULATIONS_OLLAMA,
        pop_size=10000,
        nb_agents_strat=2000,
        taille_sondage=200
    )