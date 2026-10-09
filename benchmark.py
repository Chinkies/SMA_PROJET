import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from dotenv import load_dotenv

# Import du SDK officiel Google
from google import genai

# Imports des modules de vos collègues
from population import (
    generer_population_spatiale, 
    generer_sondage, 
    fixer_aleatoire
)
from voting_systems import auditer_election

# =============================================================================
# CONFIGURATION DE L'API GEMINI
# =============================================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    client_gemini = genai.Client(api_key=API_KEY)
else:
    client_gemini = None
    print("⚠️ ATTENTION : Aucune clé GEMINI_API_KEY trouvée dans le fichier .env.")

# =============================================================================
# CAS D'ÉCOLE SPATIAL
# =============================================================================
CANDIDATS_SPATIAUX_DEFAUT = [
    {"nom": "Bob", "x": -0.45, "y": 0.10, "couleur": "#1f77b4"},
    {"nom": "Raoul", "x": -0.75, "y": 0.30, "couleur": "#d62728"},
    {"nom": "Jackie", "x": 0.00, "y": 0.00, "couleur": "#2ca02c"},
    {"nom": "Mark", "x": 0.55, "y": -0.15, "couleur": "#ff7f0e"},
    {"nom": "Francis", "x": 0.80, "y": -0.45, "couleur": "#9467bd"},
]

CLUSTERS_SPATIAUX_DEFAUT = [
    {"nom": "Bloc Gauche", "x": -0.55, "y": 0.20, "sigma": 0.18, "poids": 40.0, "couleur": "#436CAE"},
    {"nom": "Bloc Droite", "x": 0.55, "y": -0.20, "sigma": 0.18, "poids": 38.0, "couleur": "#436CAE"},
    {"nom": "Bloc Centre", "x": 0.00, "y": 0.05, "sigma": 0.14, "poids": 14.0, "couleur": "#436CAE"},
    {"nom": "Bloc Écologiste", "x": -0.10, "y": 0.60, "sigma": 0.16, "poids": 8.0, "couleur": "#436CAE"},
]

# =======================================================================
# BENCHMARK 1 & 2 : ÉVOLUTION DE LA SATISFACTION (MODÈLE SPATIAL)
# =======================================================================
def run_benchmark_courbes(nb_simulations=50, pop_size=1000, taille_sondage=None, titre=""):
    print(f"\n--- Démarrage Benchmark : {titre} ---")
    taux_values = np.linspace(0.0, 1.0, 11)
    candidats_noms = [c["nom"] for c in CANDIDATS_SPATIAUX_DEFAUT]
    
    resultats_sw = {
        "Plurality": np.zeros(11),
        "Two-Round": np.zeros(11),
        "STV": np.zeros(11),
        "Approval": np.zeros(11)
    }
    
    for sim in range(nb_simulations):
        if (sim + 1) % 10 == 0 or sim == 0:
            print(f"Simulation {sim+1}/{nb_simulations}...")
            
        current_seed = random.randint(0, 100000)
        fixer_aleatoire(current_seed)
        
        pop = generer_population_spatiale(pop_size, CANDIDATS_SPATIAUX_DEFAUT, CLUSTERS_SPATIAUX_DEFAUT)
        sondage = generer_sondage(pop, candidats_noms, taille_echantillon=taille_sondage)
        
        for i, taux in enumerate(taux_values):
            rapport = auditer_election(pop, candidats_noms, taux_strategique=taux, sondage=sondage)
            
            for sys_nom, data in rapport["resultats"].items():
                pct_satisfaction = 100.0 - data["loss_pourcentage"]
                resultats_sw[sys_nom][i] += pct_satisfaction
                
    for sys_nom in resultats_sw:
        resultats_sw[sys_nom] /= nb_simulations

    # TRACÉ
    plt.figure(figsize=(10, 6))
    couleurs = {"Plurality": "orange", "Two-Round": "red", "STV": "green", "Approval": "blue"}
    
    for sys_nom, valeurs in resultats_sw.items():
        plt.plot(taux_values * 100, valeurs, marker='o', label=sys_nom, color=couleurs[sys_nom], linewidth=2)
        
    plt.title(titre, fontsize=14, fontweight='bold')
    plt.xlabel("Proportion d'électeurs stratégiques (%)", fontsize=12)
    plt.ylabel("Satisfaction globale (% de l'Optimum)", fontsize=12)
    plt.xticks(np.arange(0, 110, 10))
    plt.ylim(bottom=min([min(v) for v in resultats_sw.values()]) - 2, top=100.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =======================================================================
# CODE DE THOMAS : WRAPPER LLM ROBUSTE (BOUCLE INFINIE ANTI-PLANTAGE)
# =======================================================================
def obtenir_vote_llm(profil_cle, utilites, sondage, systeme, candidats):
    """Interroge Gemini avec une boucle infinie pour garantir un résultat."""
    if not client_gemini:
        return list(profil_cle)[0] 
        
    prompt = f"""Tu es un électeur stratégique. Tes notes sur 100 sont : {utilites}.
Ton ordre sincère est : {list(profil_cle)}.
Le sondage donne : {sondage}. Le scrutin est : {systeme}.
Pour qui votes-tu pour maximiser ton utilité finale ? Réponds juste par le nom du candidat."""
    
    temps_attente = 5
    
    while True: # La fameuse boucle qui ne lâche rien
        try:
            reponse = client_gemini.models.generate_content(
                model='gemini-3.8-flash', 
                contents=prompt
            )
            choix = reponse.text.strip()
            
            for cand in candidats:
                if cand.lower() in choix.lower():
                    return cand
            return choix
            
        except Exception as e:
            erreur_str = str(e).lower()
            if "503" in erreur_str or "429" in erreur_str or "quota" in erreur_str or "unavailable" in erreur_str:
                print(f"    [⏳ Google saturé] Nouvelle tentative dans {temps_attente}s...")
                time.sleep(temps_attente)
                # On augmente le temps d'attente à chaque échec, plafonné à 60 secondes
                temps_attente = min(temps_attente, 10) 
            else:
                # Si c'est une autre erreur de connexion, on insiste aussi
                print(f"    [⚠️ Erreur de connexion] Réessai dans 10s... Détail : {e}")
                time.sleep(10)

def simuler_election_llm(pop, candidats, taux_strat, sondage, systeme):
    """Remplace les votes statiques par les votes du LLM pour l'échantillon."""
    nb_strat = int(len(pop) * taux_strat)
    strat_pop = pop[:nb_strat]
    
    groupes = {}
    for e in strat_pop:
        cle = tuple(e.get_classement())
        if cle not in groupes:
            groupes[cle] = e.utilities
            
    choix_llm = {}
    print(f"  -> Appels LLM en cours pour {systeme} ({len(groupes)} profils uniques)...")
    
    for i, (cle, utilites) in enumerate(groupes.items(), 1):
        print(f"     Requête {i}/{len(groupes)}...")
        choix_llm[cle] = obtenir_vote_llm(cle, utilites, sondage, systeme, candidats)
        # Pause de base de 4s pour respecter les limites du forfait gratuit
        time.sleep(4) 
        
    # Calcul du Social Welfare avec le nouveau vainqueur
    scores = {c: 0 for c in candidats}
    for e in pop[nb_strat:]:
        scores[e.get_favori()] += 1 
    for e in strat_pop:
        vote = choix_llm[tuple(e.get_classement())]
        if vote in scores:
             scores[vote] += 1 
             
    vainqueur = max(scores, key=scores.get)
    sw_obtenu = sum(e.utilities[vainqueur] for e in pop)
    
    # On renvoie la satisfaction en pourcentage de l'optimum
    optimum_candidat = max(candidats, key=lambda c: sum(e.utilities[c] for e in pop))
    sw_optimum = sum(e.utilities[optimum_candidat] for e in pop)
    
    return (sw_obtenu / sw_optimum) * 100.0

# =======================================================================
# BENCHMARK 3 : COMPARAISON STATIQUE VS LLM (Bar Chart)
# =======================================================================
def run_benchmark_llm_vs_statique():
    print("\n--- Démarrage Benchmark : LLM vs SIMPL ---")
    candidats = [c["nom"] for c in CANDIDATS_SPATIAUX_DEFAUT]
    
    # 10 000 électeurs pour stabiliser l'élection, 1% de LLM pour minimiser l'API
    pop_size = 10000
    taux_strat = 0.01 
    taille_sondage = 500
    
    pop = generer_population_spatiale(pop_size, CANDIDATS_SPATIAUX_DEFAUT, CLUSTERS_SPATIAUX_DEFAUT)
    sondage = generer_sondage(pop, candidats, taille_echantillon=taille_sondage)
    
    rapport_statique = auditer_election(pop, candidats, taux_strategique=taux_strat, sondage=sondage)
    
    sat_statique, sat_llm = [], []
    systemes = ["Plurality", "Two-Round", "STV", "Approval"]
    
    for sys in systemes:
        # Conversion de la perte en taux de satisfaction pour le Statique (SIMPL)
        sat_statique.append(100.0 - rapport_statique["resultats"][sys]["loss_pourcentage"])
        # Calcul de la satisfaction pour le LLM
        sat_llm.append(simuler_election_llm(pop, candidats, taux_strat, sondage, sys))

    # TRACÉ
    x = np.arange(len(systemes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, sat_statique, width, label='SIMPL (Statique)', color='darkorange')
    ax.bar(x + width/2, sat_llm, width, label='LLM (Gemini)', color='green')

    ax.set_ylabel('Satisfaction globale (% de l\'Optimum)', fontsize=12)
    ax.set_title('Comparaison de la Satisfaction : SIMPL vs LLM\n(Pop=10000, LLM=100, Sondage=500)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['1T (FPTP)', '2T (Two-Round)', 'SVT (STV)', 'APR (Approval)'], fontsize=11)
    
    # On zoome sur le haut du graphe pour bien voir les différences
    ax.set_ylim(bottom=min(min(sat_statique), min(sat_llm)) - 2, top=100.5)
    
    ax.legend()
    plt.tight_layout()
    plt.show()

# =======================================================================
# LANCEMENT DES TESTS
# =======================================================================
if __name__ == "__main__":
    
    NB_SIMULATIONS = 10
    """
    # Graphique 1 : Sondage Complet
    run_benchmark_courbes(
        nb_simulations=NB_SIMULATIONS, 
        pop_size=10000, 
        taille_sondage=10000, 
        titre="Évolution de la Satisfaction - Sondage Complet (100%)"
    )
    
    # Graphique 2 : Sondage Biaisé
    run_benchmark_courbes(
        nb_simulations=NB_SIMULATIONS, 
        pop_size=10000, 
        taille_sondage=200, 
        titre="Évolution de la Satisfaction - Sondage Restreint (200 / 10000)"
    )
    """
    
    # Graphique 3 : LLM vs SIMPL
    # Assurez-vous d'avoir du temps devant vous pour celui-ci !
    run_benchmark_llm_vs_statique()