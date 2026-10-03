"""
population.py - Génération de populations d'électeurs (Modèle Spatial 2D et Modèle par Factions)
et calculs de métriques de choix social.
"""

import math
import random
import numpy as np
from agent import CANDIDATS, Electeur


# =============================================================================
# REPRODUCTIBILITÉ
# =============================================================================
def fixer_aleatoire(seed_value=42):
    """Fixe la graine aléatoire pour Python et NumPy."""
    np.random.seed(seed_value)
    random.seed(seed_value)


# =============================================================================
# 1. GÉNÉRATEUR SPATIAL 2D
# =============================================================================
def generer_population_spatiale(nb_electeurs, candidats_spatiaux, clusters_spatiaux):
    """
    Génère une population d'agents positionnés sur l'échiquier politique 2D [-1, 1]^2.
    
    L'utilité est inversement proportionnelle à la distance euclidienne :
    u(e, c) = 100 * (1 - dist / d_max), où d_max = 2 * sqrt(2).
    """
    d_max = 2.0 * math.sqrt(2.0)
    population = []

    # Normalisation des poids des clusters
    poids_bruts = np.array([float(cl["poids"]) for cl in clusters_spatiaux], dtype=float)
    probas_clusters = poids_bruts / np.sum(poids_bruts)
    indices_clusters = np.arange(len(clusters_spatiaux))

    for i in range(nb_electeurs):
        # 1. Sélection du cluster
        idx_cl = np.random.choice(indices_clusters, p=probas_clusters)
        cluster = clusters_spatiaux[idx_cl]

        # 2. Tirage de la position (x, y) selon la dispersion du cluster
        x_e = np.random.normal(cluster["x"], cluster["sigma"])
        y_e = np.random.normal(cluster["y"], cluster["sigma"])
        x_e = float(np.clip(x_e, -1.0, 1.0))
        y_e = float(np.clip(y_e, -1.0, 1.0))

        # 3. Calcul des utilités spatiales pour chaque candidat
        utilities = {}
        for c in candidats_spatiaux:
            dist = math.hypot(x_e - c["x"], y_e - c["y"])
            score = 100.0 * (1.0 - (dist / d_max))
            utilities[c["nom"]] = round(float(np.clip(score, 0.0, 100.0)), 1)

        electeur = Electeur(
            id_electeur=i,
            utilities=utilities,
            faction=cluster["nom"],
            position=(round(x_e, 3), round(y_e, 3))
        )
        population.append(electeur)

    return population


# =============================================================================
# 2. GÉNÉRATEURS PAR FACTIONS
# =============================================================================
def generer_population_factions_fixe(nb_electeurs):
    """Génère une population avec des factions et lois normales prédéfinies."""
    population = []
    noms_factions = ["Pro-Bob", "Pro-Raoul", "Pro-Jackie", "Pro-Mark", "Pro-Francis"]
    probas_factions = [0.35, 0.25, 0.20, 0.12, 0.08]

    for i in range(nb_electeurs):
        faction = np.random.choice(noms_factions, p=probas_factions)

        if faction == "Pro-Bob":
            score_bob = np.random.normal(85, 10)
            score_raoul = np.random.normal(45, 15)
            score_jackie = np.random.normal(15, 10)
            score_mark = np.random.normal(40, 20)
            score_francis = np.random.normal(10, 5)
        elif faction == "Pro-Raoul":
            score_bob = np.random.normal(30, 15)
            score_raoul = np.random.normal(85, 10)
            score_jackie = np.random.normal(40, 15)
            score_mark = np.random.normal(50, 15)
            score_francis = np.random.normal(20, 10)
        elif faction == "Pro-Jackie":
            score_bob = np.random.normal(15, 10)
            score_raoul = np.random.normal(40, 15)
            score_jackie = np.random.normal(85, 10)
            score_mark = np.random.normal(60, 15)
            score_francis = np.random.normal(30, 15)
        elif faction == "Pro-Mark":
            score_bob = np.random.normal(40, 15)
            score_raoul = np.random.normal(50, 15)
            score_jackie = np.random.normal(60, 15)
            score_mark = np.random.normal(85, 10)
            score_francis = np.random.normal(70, 15)
        else:  # Pro-Francis
            score_bob = np.random.normal(10, 10)
            score_raoul = np.random.normal(20, 15)
            score_jackie = np.random.normal(30, 15)
            score_mark = np.random.normal(70, 15)
            score_francis = np.random.normal(85, 10)

        utilities = {
            "Bob": round(float(np.clip(score_bob, 0, 100)), 1),
            "Raoul": round(float(np.clip(score_raoul, 0, 100)), 1),
            "Jackie": round(float(np.clip(score_jackie, 0, 100)), 1),
            "Mark": round(float(np.clip(score_mark, 0, 100)), 1),
            "Francis": round(float(np.clip(score_francis, 0, 100)), 1)
        }

        electeur = Electeur(id_electeur=i, utilities=utilities, faction=faction)
        population.append(electeur)

    return population


def generer_population_factions_aleatoire(nb_electeurs):
    """Génère une population aléatoire via distribution de Dirichlet."""
    noms_candidats = list(CANDIDATS.values())
    nb_candidats = len(noms_candidats)

    probabilites_factions = np.random.dirichlet(np.ones(nb_candidats))

    matrice_moyennes = []
    for i in range(nb_candidats):
        moyennes_faction = []
        for j in range(nb_candidats):
            if i == j:
                moyennes_faction.append(np.random.uniform(70, 100))
            else:
                moyennes_faction.append(np.random.uniform(0, 70))
        matrice_moyennes.append(moyennes_faction)

    population = []
    indices_factions = np.arange(nb_candidats)

    for i in range(nb_electeurs):
        idx = np.random.choice(indices_factions, p=probabilites_factions)
        moyennes = matrice_moyennes[idx]
        nom_faction = f"Pro-{noms_candidats[idx]}"

        utilities = {}
        for j, nom in enumerate(noms_candidats):
            ecart_type = np.random.uniform(10, 20)
            score = np.random.normal(moyennes[j], ecart_type)
            utilities[nom] = round(float(np.clip(score, 0, 100)), 1)

        electeur = Electeur(id_electeur=i, utilities=utilities, faction=nom_faction)
        population.append(electeur)

    return population, probabilites_factions, matrice_moyennes


# =============================================================================
# SONDAGE & MÉTRIQUES SOCIALES
# =============================================================================
def generer_sondage(population, liste_candidats, taille_echantillon=None):
    """Réalise un sondage d'intentions de vote (1er choix sincère)."""
    if taille_echantillon is None or taille_echantillon >= len(population):
        sondes = population
    else:
        sondes = random.sample(population, taille_echantillon)

    intentions = {c: 0 for c in liste_candidats}
    for electeur in sondes:
        intentions[electeur.get_favori()] += 1

    total_sondes = len(sondes)
    return {c: round((v / total_sondes) * 100, 1) for c, v in intentions.items()}


def calculer_optimum_social(population, liste_candidats):
    """Calcule le candidat socialement optimal et le bien-être social total."""
    scores_totaux = {candidat: 0.0 for candidat in liste_candidats}

    for electeur in population:
        for candidat in liste_candidats:
            scores_totaux[candidat] += electeur.utilities.get(candidat, 0.0)

    gagnant_optimal = max(scores_totaux, key=scores_totaux.get)
    score_max = scores_totaux[gagnant_optimal]
    return gagnant_optimal, score_max, scores_totaux