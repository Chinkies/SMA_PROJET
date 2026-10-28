"""
llm_agents.py - Gestion des agents stratégiques basés sur un LLM local via Ollama
(avec auto-détection du modèle installé et moteur cognitif d'utilité espérée).
"""

import math
import os
import random
import re
import requests

from agent import generer_bulletin_statique
from population import generer_sondage

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"


def detecter_modele_ollama():
    """
    Interroge automatiquement le démon local Ollama pour trouver un modèle déjà installé.
    Évite l'erreur 404 si l'utilisateur a installé 'mistral', 'llama3', 'qwen', etc.
    """
    modele_env = os.getenv("OLLAMA_MODEL")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            modeles_dispos = [m["name"] for m in resp.json().get("models", [])]
            if modeles_dispos:
                if modele_env and any(modele_env in m for m in modeles_dispos):
                    return modele_env, True
                # Priorité aux modèles rapides s'ils sont présents
                for pref in ["llama3.2", "llama3", "mistral", "qwen", "gemma", "phi"]:
                    for m in modeles_dispos:
                        if pref in m.lower():
                            return m, True
                return modeles_dispos[0], True
    except Exception:
        pass
    return (modele_env or "llama3.2"), False


OLLAMA_MODEL, OLLAMA_DISPONIBLE = detecter_modele_ollama()
if OLLAMA_DISPONIBLE:
    print(f"[🤖 Ollama détecté] Modèle actif sélectionné automatiquement : '{OLLAMA_MODEL}'")
else:
    print(
        f"[ℹ️ Info Ollama] Aucun modèle trouvé dans Ollama sur {OLLAMA_BASE_URL}.\n"
        f"   -> Astuce : Lancez 'ollama pull llama3.2' dans votre terminal pour utiliser le serveur Ollama.\n"
        f"   -> En attendant, l'agent IA utilisera le moteur cognitif d'Utilité Espérée (Myerson-Weber)."
    )


# =============================================================================
# RAISONNEMENT COGNITIF IA (EXPECTED UTILITY) & PROMPTS OLLAMA
# =============================================================================
def decision_ia_utilite_esperee(utilities, classement_sincere, sondage, systeme_vote):
    """
    Modèle de décision rationnelle avancée (Myerson-Weber / Quantal Response) :
    Contrairement à l'agent statique rigide qui ne regarde qu'un seuil fixe (Top 2),
    un agent IA évalue la probabilité continue de pivot de chaque candidat dans le sondage
    pondérée par le gain marginal d'utilité espérée.
    """
    candidats = list(utilities.keys())
    max_sond = max(sondage.values()) if sondage else 20.0

    # Probabilité perçue de victoire/qualification (Softmax sur les intentions du sondage)
    poids_viabilite = {
        c: math.exp((sondage.get(c, 0.0) - max_sond) / 6.5)
        for c in candidats
    }
    somme_p = sum(poids_viabilite.values())
    probas = {c: poids_viabilite[c] / somme_p for c in candidats}

    # Utilité espérée du scrutin sans intervention
    eu_reference = sum(probas[c] * utilities[c] for c in candidats)

    if "Plurality" in systeme_vote:
        # L'IA maximise : P(candidat est pivot) * (u(c) - EU_reference) + attachement sincère
        scores_ia = {
            c: probas[c] * (utilities[c] - eu_reference) + 0.18 * utilities[c]
            for c in candidats
        }
        return max(scores_ia, key=scores_ia.get)

    elif "Two-Round" in systeme_vote:
        # L'IA anticipe le 2nd tour : soutient le candidat capable de se qualifier ET de gagner son duel
        scores_t1 = {
            c: (probas[c] ** 0.7) * (utilities[c] - eu_reference * 0.85) + 0.25 * utilities[c]
            for c in candidats
        }
        return max(scores_t1, key=scores_t1.get)

    elif "STV" in systeme_vote:
        # L'IA préserve les transferts de voix sincères vers le compromis (contrairement à l'agent
        # statique qui détruit le compromis par un burying aveugle) et enterre uniquement le pire rival viable
        top_2 = sorted(sondage, key=sondage.get, reverse=True)[:2]
        pire_leader = min(top_2, key=lambda c: utilities[c])
        if utilities[pire_leader] < eu_reference:
            bulletin = [c for c in classement_sincere if c != pire_leader] + [pire_leader]
            return bulletin
        return list(classement_sincere)

    else:  # Approval Voting
        # Règle optimale de laslier / Myerson-Weber : approuver tout candidat dont l'utilité
        # dépasse l'utilité espérée pondérée par les probabilités du sondage !
        approuves = [c for c in candidats if utilities[c] >= eu_reference + 2.0]
        if not approuves:
            approuves = [classement_sincere[0]]
        return approuves


def construire_prompt_electeur(utilities, classement_sincere, sondage_actuel, systeme_vote, liste_candidats):
    """Construit le prompt stratégique envoyé à Ollama."""
    if "Plurality" in systeme_vote:
        consigne = (
            "Scrutin : Majoritaire à 1 tour (Plurality).\n"
            "Objectif : Maximiser ton espérance d'utilité en arbitrant entre ton favori et les candidats les mieux placés du sondage.\n"
            f"Réponds UNIQUEMENT par le nom exact du candidat parmi {liste_candidats}, sans aucune phrase."
        )
    elif "Two-Round" in systeme_vote:
        consigne = (
            "Scrutin : 1er tour d'une élection à 2 tours (les 2 premiers vont en finale).\n"
            "Objectif : Qualifier le meilleur candidat capable de l'emporter au second tour.\n"
            f"Réponds UNIQUEMENT par le nom exact du candidat parmi {liste_candidats}, sans aucune phrase."
        )
    elif "STV" in systeme_vote:
        consigne = (
            "Scrutin : Vote alternatif par classement (STV). Le dernier est éliminé et ses voix sont transférées.\n"
            "Objectif : Ordonne les 5 candidats pour maximiser l'utilité du vainqueur final grâce aux transferts.\n"
            f"Réponds UNIQUEMENT par les 5 noms séparés par des virgules parmi {liste_candidats}."
        )
    else:  # Approval
        consigne = (
            "Scrutin : Vote par approbation (Approval Voting).\n"
            "Objectif : Approuve tous les candidats qui t'apportent une utilité supérieure à l'espérance du sondage (ton favori + les compromis acceptables).\n"
            f"Réponds UNIQUEMENT par les noms approuvés séparés par des virgules parmi {liste_candidats}."
        )

    return (
        f"Tu es un agent IA expert en théorie des jeux votant à une élection.\n"
        f"- Tes utilités (sur 100) : {utilities}\n"
        f"- Ton classement sincère : {list(classement_sincere)}\n"
        f"- Sondage d'intentions de vote (%) : {sondage_actuel}\n"
        f"{consigne}\n"
        f"Réponse (aucune explication, juste le(s) nom(s) parmi {liste_candidats}) :"
    )


def preparer_prompts_llm(echantillon_strategique, sondage_actuel, systeme_vote):
    """Regroupe les électeurs stratégiques par ordre de préférence exact."""
    groupes_profils = {}
    liste_candidats = list(echantillon_strategique[0].utilities.keys()) if echantillon_strategique else []

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
        prompts_a_envoyer[profil_cle] = construire_prompt_electeur(
            electeur_type.utilities,
            profil_cle,
            sondage_actuel,
            systeme_vote,
            liste_candidats
        )

    return groupes_profils, prompts_a_envoyer


def parser_reponse_llm(texte_brut, liste_candidats, systeme_vote, classement_sincere):
    """Extrait le vote (candidat unique ou liste) depuis la réponse d'Ollama."""
    trouves = []
    pattern = "|".join(re.escape(c) for c in liste_candidats)
    for match in re.finditer(pattern, texte_brut, flags=re.IGNORECASE):
        mot = match.group(0).lower()
        for c in liste_candidats:
            if c.lower() == mot and c not in trouves:
                trouves.append(c)

    if "STV" in systeme_vote:
        for c in classement_sincere:
            if c not in trouves:
                trouves.append(c)
        return trouves
    elif "Approval" in systeme_vote:
        return trouves if trouves else [classement_sincere[0]]
    else:
        return trouves[0] if trouves else classement_sincere[0]


def interroger_ollama(prompt, liste_candidats, systeme_vote="Plurality", classement_sincere=None, utilities=None, sondage=None):
    """
    Interroge le modèle Ollama détecté. Si Ollama renvoie 404 (modèle non téléchargé)
    ou est éteint, bascule automatiquement sur le moteur cognitif d'Utilité Espérée.
    """
    global OLLAMA_DISPONIBLE
    if classement_sincere is None:
        classement_sincere = liste_candidats

    if OLLAMA_DISPONIBLE:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 30}
        }
        try:
            resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=20)
            if resp.status_code == 200:
                texte = resp.json().get("response", "").strip()
                return parser_reponse_llm(texte, liste_candidats, systeme_vote, classement_sincere)
            else:
                print(f"  [ℹ️ Modèle '{OLLAMA_MODEL}' absent d'Ollama (HTTP {resp.status_code}) -> Bascule sur Moteur IA d'Utilité Espérée]")
                OLLAMA_DISPONIBLE = False
        except Exception:
            OLLAMA_DISPONIBLE = False

    # Raisonnement IA de secours (Utilité Espérée de Myerson-Weber) si Ollama n'a pas de modèle
    if utilities and sondage:
        return decision_ia_utilite_esperee(utilities, classement_sincere, sondage, systeme_vote)
    return classement_sincere[0]


def executer_mode_llm(groupes, prompts, liste_candidats, systeme_vote="Plurality", sondage_actuel=None):
    """Exécute la prise de décision IA pour chaque groupe de profils."""
    reponses_strategiques = {}
    for profil_cle, prompt in prompts.items():
        electeur_ex = groupes[profil_cle]["exemple_electeur"]
        choix_llm = interroger_ollama(
            prompt,
            liste_candidats,
            systeme_vote=systeme_vote,
            classement_sincere=list(profil_cle),
            utilities=electeur_ex.utilities,
            sondage=sondage_actuel
        )
        reponses_strategiques[profil_cle] = choix_llm

    return reponses_strategiques


def preparer_election_mixte(population, liste_candidats, pct_strategique, systeme_vote, taille_echantillon_sondage, mode_llm=False):
    """Orchestre une élection mixte (sincères + stratégiques Statiques ou IA)."""
    nb_strat = int(len(population) * (pct_strategique / 100.0))
    pop_melangee = population.copy()
    random.shuffle(pop_melangee)

    groupe_strategique = pop_melangee[:nb_strat]
    groupe_sincere = pop_melangee[nb_strat:]

    sondage = generer_sondage(population, liste_candidats, taille_echantillon=taille_echantillon_sondage)
    bulletins_finaux = []

    for electeur in groupe_sincere:
        bulletins_finaux.append(
            generer_bulletin_statique(electeur, systeme_vote, strategique=False, sondage=None)
        )

    if not mode_llm:
        for electeur in groupe_strategique:
            bulletins_finaux.append(
                generer_bulletin_statique(electeur, systeme_vote, strategique=True, sondage=sondage)
            )
    else:
        groupes, prompts = preparer_prompts_llm(groupe_strategique, sondage, systeme_vote)
        choix_par_profil = executer_mode_llm(
            groupes, prompts, liste_candidats, systeme_vote=systeme_vote, sondage_actuel=sondage
        )

        for profil_cle, data in groupes.items():
            vote_choisi = choix_par_profil[profil_cle]
            for electeur in data["membres"]:
                if vote_choisi is None:
                    vote_choisi = generer_bulletin_statique(electeur, systeme_vote, strategique=False, sondage=None)
                bulletins_finaux.append(vote_choisi)

    return bulletins_finaux, sondage