"""
population.py - Génération de populations d'électeurs (Modèle Spatial 2D réaliste,
Générateur Spatial Aléatoire avec effet Spoiler/Condorcet et Modèle par Factions).
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
    Utilise une décroissance spatiale concave permettant au candidat centriste (Condorcet)
    d'être l'Optimum Social global tout en étant minoritaire en 1er choix (Squeeze effect).
    """
    population = []

    poids_bruts = np.array([float(cl["poids"]) for cl in clusters_spatiaux], dtype=float)
    probas_clusters = poids_bruts / np.sum(poids_bruts)
    indices_clusters = np.arange(len(clusters_spatiaux))

    for i in range(nb_electeurs):
        idx_cl = np.random.choice(indices_clusters, p=probas_clusters)
        cluster = clusters_spatiaux[idx_cl]

        x_e = float(np.clip(np.random.normal(cluster["x"], cluster["sigma"]), -1.0, 1.0))
        y_e = float(np.clip(np.random.normal(cluster["y"], cluster["sigma"]), -1.0, 1.0))

        utilities = {}
        for c in candidats_spatiaux:
            dist = math.hypot(x_e - c["x"], y_e - c["y"])
            # Décroissance quadratique/gaussienne : pénalise fortement les candidats
            # du camp opposé (dist > 1.0) et valorise le compromis centriste en 2e choix
            score_base = 100.0 * math.exp(-(dist ** 2) / (2.0 * (0.68 ** 2)))
            bruit_indiv = np.random.normal(0.0, 1.2)
            utilities[c["nom"]] = round(float(np.clip(score_base + bruit_indiv, 0.0, 100.0)), 1)

        electeur = Electeur(
            id_electeur=i,
            utilities=utilities,
            faction=cluster["nom"],
            position=(round(x_e, 3), round(y_e, 3))
        )
        population.append(electeur)

    return population


def generer_election_spatiale_aleatoire(nb_electeurs):
    """
    Génère une élection spatiale 2D réaliste illustrant :
      1. L'effet Spoiler (division des voix dans le camp majoritaire entre 2 candidats proches).
      2. Le Center-Squeeze Effect (le candidat centriste Jackie est l'Optimum Social / Condorcet,
         mais a peu de 1ers choix sincères car pris en étau entre les grands blocs).
    """
    # Choix aléatoire du camp majoritaire mais divisé (Gauche ou Droite)
    inversion = random.choice([-1.0, 1.0])

    # Poids des blocs :
    # - Camp A (Majoritaire mais divisé entre 2 candidats) : ~46% à 50%
    # - Camp B (Minoritaire mais uni derrière 1 grand candidat) : ~34% à 38%
    # - Centre (Modérés pro-compromis) : ~10% à 14%
    # - Pôle contestataire/radical : ~5% à 8%
    p_camp_divise = float(np.random.uniform(46.0, 50.0))
    p_camp_uni = float(np.random.uniform(34.0, 38.0))
    p_centre = float(np.random.uniform(10.0, 13.0))
    p_autre = max(4.0, 100.0 - (p_camp_divise + p_camp_uni + p_centre))

    clusters_alea = [
        {
            "nom": "Bloc Majoritaire Divisé",
            "x": inversion * float(np.random.uniform(-0.55, -0.48)),
            "y": float(np.random.uniform(0.05, 0.15)),
            "sigma": float(np.random.uniform(0.18, 0.22)),
            "poids": p_camp_divise,
            "couleur": "#436CAE"
        },
        {
            "nom": "Bloc Opposant Uni",
            "x": inversion * float(np.random.uniform(0.52, 0.62)),
            "y": float(np.random.uniform(-0.15, -0.05)),
            "sigma": float(np.random.uniform(0.16, 0.20)),
            "poids": p_camp_uni,
            "couleur": "#d62728"
        },
        {
            "nom": "Bloc Central",
            "x": float(np.random.uniform(-0.05, 0.05)),
            "y": float(np.random.uniform(-0.05, 0.05)),
            "sigma": float(np.random.uniform(0.12, 0.16)),
            "poids": p_centre,
            "couleur": "#2ca02c"
        },
        {
            "nom": "Bloc Radical",
            "x": inversion * float(np.random.uniform(0.75, 0.85)),
            "y": float(np.random.uniform(-0.45, -0.30)),
            "sigma": float(np.random.uniform(0.14, 0.18)),
            "poids": p_autre,
            "couleur": "#9467bd"
        },
    ]

    # Positionnement des 5 candidats :
    # - Jackie : Centriste (Optimum Social / Condorcet), aimée de tous en 2e choix mais ~13% au 1er tour
    # - Bob (Centre-Gauche) & Raoul (Gauche) : Se divisent le Bloc Majoritaire (~23% et ~22% chacun)
    # - Mark : Candidat du Bloc Opposant Uni (~33% au 1er tour -> Gagne en Plurality sincère malgré son rejet par la majorité !)
    # - Francis : Candidat extrême (~9%)
    candidats_alea = [
        {
            "nom": "Jackie",  # Candidat de compromis (Optimum Social)
            "x": float(np.random.uniform(-0.06, 0.06)),
            "y": float(np.random.uniform(-0.04, 0.06)),
            "couleur": "#2ca02c"
        },
        {
            "nom": "Bob",  # Modéré du camp majoritaire (2e meilleur bien-être social)
            "x": inversion * float(np.random.uniform(-0.38, -0.32)),
            "y": float(np.random.uniform(0.02, 0.10)),
            "couleur": "#1f77b4"
        },
        {
            "nom": "Raoul",  # Concurrent interne (Spoiler) du camp majoritaire
            "x": inversion * float(np.random.uniform(-0.68, -0.60)),
            "y": float(np.random.uniform(0.12, 0.22)),
            "couleur": "#d62728"
        },
        {
            "nom": "Mark",  # Leader du camp minoritaire uni (Gagnant FPTP sincère par division adverse)
            "x": inversion * float(np.random.uniform(0.52, 0.60)),
            "y": float(np.random.uniform(-0.15, -0.05)),
            "couleur": "#ff7f0e"
        },
        {
            "nom": "Francis",  # Extrême
            "x": inversion * float(np.random.uniform(0.82, 0.92)),
            "y": float(np.random.uniform(-0.45, -0.35)),
            "couleur": "#9467bd"
        },
    ]

    pop = generer_population_spatiale(nb_electeurs, candidats_alea, clusters_alea)
    return pop, candidats_alea, clusters_alea


# =============================================================================
# 2. GÉNÉRATEURS PAR FACTIONS (COMPATIBILITÉ APP STREAMLIT)
# =============================================================================
def generer_population_factions_fixe(nb_electeurs):
    """Génère une population avec des factions et lois normales prédéfinies."""
    population = []
    noms_factions = ["Pro-Bob", "Pro-Raoul", "Pro-Jackie", "Pro-Mark", "Pro-Francis"]
    probas_factions = [0.35, 0.25, 0.20, 0.12, 0.08]

    for i in range(nb_electeurs):
        faction = np.random.choice(noms_factions, p=probas_factions)

        if faction == "Pro-Bob":
            scores = [np.random.normal(85, 10), np.random.normal(45, 15), np.random.normal(55, 12), np.random.normal(30, 15), np.random.normal(10, 5)]
        elif faction == "Pro-Raoul":
            scores = [np.random.normal(40, 15), np.random.normal(85, 10), np.random.normal(45, 15), np.random.normal(25, 15), np.random.normal(15, 10)]
        elif faction == "Pro-Jackie":
            scores = [np.random.normal(50, 12), np.random.normal(40, 15), np.random.normal(88, 8), np.random.normal(55, 12), np.random.normal(25, 15)]
        elif faction == "Pro-Mark":
            scores = [np.random.normal(30, 15), np.random.normal(25, 15), np.random.normal(60, 12), np.random.normal(85, 10), np.random.normal(65, 15)]
        else:
            scores = [np.random.normal(10, 10), np.random.normal(15, 15), np.random.normal(35, 15), np.random.normal(65, 15), np.random.normal(85, 10)]

        noms = ["Bob", "Raoul", "Jackie", "Mark", "Francis"]
        utilities = {nom: round(float(np.clip(s, 0, 100)), 1) for nom, s in zip(noms, scores)}
        population.append(Electeur(id_electeur=i, utilities=utilities, faction=faction))

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
                moyennes_faction.append(np.random.uniform(75, 95))
            else:
                moyennes_faction.append(np.random.uniform(10, 68))
        matrice_moyennes.append(moyennes_faction)

    population = []
    indices_factions = np.arange(nb_candidats)

    for i in range(nb_electeurs):
        idx = np.random.choice(indices_factions, p=probabilites_factions)
        moyennes = matrice_moyennes[idx]
        nom_faction = f"Pro-{noms_candidats[idx]}"

        utilities = {}
        for j, nom in enumerate(noms_candidats):
            ecart_type = np.random.uniform(10, 18)
            score = np.random.normal(moyennes[j], ecart_type)
            utilities[nom] = round(float(np.clip(score, 0, 100)), 1)

        population.append(Electeur(id_electeur=i, utilities=utilities, faction=nom_faction))

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
    """Calcule le candidat socialement optimal et les scores de bien-être social."""
    scores_totaux = {candidat: 0.0 for candidat in liste_candidats}

    for electeur in population:
        for candidat in liste_candidats:
            scores_totaux[candidat] += electeur.utilities.get(candidat, 0.0)

    gagnant_optimal = max(scores_totaux, key=scores_totaux.get)
    score_max = scores_totaux[gagnant_optimal]
    return gagnant_optimal, score_max, scores_totaux