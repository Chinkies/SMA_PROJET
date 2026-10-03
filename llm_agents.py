"""
llm_agents.py - Gestion des agents stratégiques basés sur un LLM (Google Gemini)
et orchestration de l'élection mixte.
"""

import os
import random
import time
from dotenv import load_dotenv
from google import genai

from agent import generer_bulletin_statique
from population import generer_sondage

load_dotenv()

# Initialisation du client Gemini
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# =============================================================================
# REGROUPEMENT ET GÉNÉRATION DES PROMPTS
# =============================================================================
def preparer_prompts_llm(echantillon_strategique, sondage_actuel, systeme_vote):
    """
    Regroupe les électeurs stratégiques par ordre de préférence exact
    pour minimiser le nombre d'appels API.
    """
    groupes_profils = {}

    for electeur in echantillon_strategique:
        profil_cle = tuple(electeur.get_classement())

        if profil_cle not in groupes_profils:
            groupes_profils[profil_cle] = {
                "exemple_electeur": electeur,
                "membres": []
            }
        groupes_profils[profil_cle]["membres"].append(electeur)

    prompts_a_envoyer = {}
    for profil_cle, data in groupes_profils.items():
        electeur_type = data["exemple_electeur"]

        prompt = f"""Tu es un citoyen votant de manière stratégique. 
Tes notes de préférence pour les candidats (sur 100) sont : {electeur_type.utilities}
Ton classement sincère est : {list(profil_cle)}

Le dernier sondage donne ces intentions de vote : {sondage_actuel}
Le mode de scrutin actuel est : {systeme_vote}.

Sachant que tu veux maximiser ton utilité finale et éviter l'élection des candidats que tu détestes, quel est ton vote stratégique ?
Réponds uniquement par le nom du candidat (ou la liste ordonnée/sous-liste selon le scrutin), sans aucune autre phrase."""

        prompts_a_envoyer[profil_cle] = prompt

    return groupes_profils, prompts_a_envoyer


# =============================================================================
# APPELS API GEMINI
# =============================================================================
def interroger_gemini(prompt, liste_candidats, max_tentatives=3):
    """
    Envoie le prompt à Gemini avec temporisation et réessais.
    Nettoie la réponse pour garantir un candidat ou une sélection valide.
    """
    for tentative in range(max_tentatives):
        try:
            reponse = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            choix = reponse.text.strip()

            # Nettoyage pour les scrutins à choix unique
            for candidat in liste_candidats:
                if candidat.lower() in choix.lower():
                    return candidat

            return choix

        except Exception as e:
            erreur_str = str(e)
            if "503" in erreur_str or "429" in erreur_str:
                temps_attente = (tentative + 1) * 5
                print(f"\n[⚠️ Serveur Google occupé] Nouvelle tentative dans {temps_attente} secondes...")
                time.sleep(temps_attente)
            else:
                print(f"Erreur API Gemini inattendue : {e}")
                return None

    print("\n[❌ Échec] Impossible d'obtenir une réponse de Gemini après plusieurs tentatives.")
    return None


def executer_mode_llm(groupes, prompts, liste_candidats):
    """
    Interroge le LLM une seule fois par profil distinct et distribue la réponse
    à tous les électeurs partageant ce profil.
    """
    reponses_strategiques = {}
    total_requetes = len(prompts)

    print(f"Lancement de {total_requetes} requêtes vers Gemini (Mode LLM)...")

    for i, (profil_cle, prompt) in enumerate(prompts.items(), 1):
        print(f"Requête {i}/{total_requetes} en cours...")
        choix_llm = interroger_gemini(prompt, liste_candidats)
        reponses_strategiques[profil_cle] = choix_llm

        if i < total_requetes:
            time.sleep(4)

    return reponses_strategiques


# =============================================================================
# ORCHESTRATION DE L'ÉLECTION MIXTE
# =============================================================================
def preparer_election_mixte(population, liste_candidats, pct_strategique, systeme_vote, taille_echantillon_sondage, mode_llm=False):
    """
    Sépare la population entre sincères et stratégiques, génère le sondage,
    et produit la liste complète des bulletins dépouillables.
    """
    nb_strat = int(len(population) * (pct_strategique / 100.0))
    pop_melangee = population.copy()
    random.shuffle(pop_melangee)

    groupe_strategique = pop_melangee[:nb_strat]
    groupe_sincere = pop_melangee[nb_strat:]

    sondage = generer_sondage(population, liste_candidats, taille_echantillon=taille_echantillon_sondage)
    bulletins_finaux = []

    # 1. Émission des bulletins sincères
    for electeur in groupe_sincere:
        bulletins_finaux.append(
            generer_bulletin_statique(electeur, systeme_vote, strategique=False, sondage=None)
        )

    # 2. Émission des bulletins stratégiques
    if not mode_llm:
        for electeur in groupe_strategique:
            bulletins_finaux.append(
                generer_bulletin_statique(electeur, systeme_vote, strategique=True, sondage=sondage)
            )
    else:
        groupes, prompts = preparer_prompts_llm(groupe_strategique, sondage, systeme_vote)
        choix_par_profil = executer_mode_llm(groupes, prompts, liste_candidats)

        for profil_cle, data in groupes.items():
            vote_choisi = choix_par_profil[profil_cle]

            for electeur in data["membres"]:
                # Repli sur le bulletin sincère en cas d'échec API
                if vote_choisi is None:
                    vote_choisi = generer_bulletin_statique(electeur, systeme_vote, strategique=False, sondage=None)

                # Formatage STV ou Approval : conversion texte -> liste si nécessaire
                if systeme_vote in ["STV / Ranked-Choice", "Approval Voting"] and isinstance(vote_choisi, str):
                    vote_choisi = [c.strip() for c in vote_choisi.split(",") if c.strip() in liste_candidats]

                bulletins_finaux.append(vote_choisi)

    return bulletins_finaux, sondage