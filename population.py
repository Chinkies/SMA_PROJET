import random
from collections import Counter
import numpy as np
import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()
# Avec la nouvelle librairie, on initialise un "Client"
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#=====================================================================================================
#======================================CANDIDATS======================================================
#=====================================================================================================

CANDIDATS = {0: "Bob", 1: "Raoul", 2: "Jackie", 3: "Mark", 4: "Francis"}

#=====================================================================================================
#======================================ELECTEURS======================================================
#=====================================================================================================

class Electeur:
    def __init__(self, id_electeur, utilities, faction="Inconnue"):
        self.id = id_electeur
        self.utilities = utilities
        self.faction = faction

    def get_favori(self):
        return max(self.utilities, key=self.utilities.get)

    def get_classement(self):
        return sorted(self.utilities, key=self.utilities.get, reverse=True)

    def get_approbations(self, seuil=50):
        approbations = [candidat for candidat, score in self.utilities.items() if score >= seuil]
        if not approbations:
            approbations = [self.get_favori()]
        return approbations

    #-------------------------------------------------------------------------------------------------
    #---------------------------STRATÉGIES DE VOTE PAR MODE DE SCRUTIN--------------------------------
    #-------------------------------------------------------------------------------------------------

    def voter_pluralite(self, strategique=False, sondage=None):
        """
        Plurality / FPTP
        - Sincère : vote pour le candidat favori.
        - Stratégique (Compromis) : si le favori n'est pas dans les 2 leaders
          du sondage, vote pour le préféré parmi les 2 leaders (vote utile).
        """
        favori = self.get_favori()
        if not strategique or not sondage:
            return favori

        top_2 = sorted(sondage, key=sondage.get, reverse=True)[:2]

        if favori in top_2:
            return favori

        cand_a, cand_b = top_2[0], top_2[1]
        return cand_a if self.utilities[cand_a] >= self.utilities[cand_b] else cand_b

    def voter_deux_tours_t1(self, strategique=False, sondage=None):
        """
        Two-Round Runoff Voting (1er tour)
        - Sincère : vote pour le favori.
        - Stratégique : sécurise sa voix en votant pour le meilleur candidat
          parmi les 3 plus grands prétendants au second tour.
        """
        # Attention, il existe une autre comportement pour le vote stratégique deux tours : 
        # le pari stratégique, voter pour un candidat adverse plus faible au 1er tour afin 
        # d'assurer un duel facile au 2d tour pour son propre favori.

        favori = self.get_favori()
        if not strategique or not sondage:
            return favori

        viables = sorted(sondage, key=sondage.get, reverse=True)[:3]

        if favori in viables:
            return favori

        return max(viables, key=lambda c: self.utilities[c])

    def voter_stv(self, strategique=False, sondage=None):
        """
        Single Transferable Vote (STV)
        - Sincère : classement par ordre d'utilité décroissante.
        - Stratégique (Burying) : relègue le principal rival du favori à la 
          dernière position de son bulletin pour limiter ses reports de voix.
        """
        classement_sincere = self.get_classement()
        if not strategique or not sondage:
            return classement_sincere

        favori = classement_sincere[0]
        sondage_hors_favori = {c: p for c, p in sondage.items() if c != favori}
        rival_menacant = max(sondage_hors_favori, key=sondage_hors_favori.get)

        bulletin_modifie = [c for c in classement_sincere if c != rival_menacant]
        bulletin_modifie.append(rival_menacant)
        return bulletin_modifie

    def voter_approbation(self, strategique=False, sondage=None, seuil_defaut=50):
        """
        Approval Voting
        - Sincère : approuve tout candidat avec u_i(c) >= seuil_defaut.
        - Stratégique (Threshold Adaptation / Bullet Voting) :
          Fixe le seuil d'approbation au niveau d'utilité du leader qu'il préfère
          dans le duel au sommet pour maximiser son impact sur la victoire.
        """
        if not strategique or not sondage:
            return self.get_approbations(seuil=seuil_defaut)

        leaders = sorted(sondage, key=sondage.get, reverse=True)[:2]
        leader_prefere = max(leaders, key=lambda c: self.utilities[c])
        seuil_strategique = self.utilities[leader_prefere]

        return [c for c, u in self.utilities.items() if u >= seuil_strategique]


#=====================================================================================================
#========================GUIDE POUR QUE L'ÉLECTEUR CHOISISE LA BONNE MÉTHODE==========================
#=====================================================================================================

def obtenir_vote_statique(electeur, systeme_vote, strategique, sondage):
    """Aiguille l'électeur vers la bonne méthode de sa classe selon le système."""
    if systeme_vote == "Plurality (FPTP)":
        return electeur.voter_pluralite(strategique, sondage)
    elif systeme_vote == "Two-Round Runoff":
        return electeur.voter_deux_tours_t1(strategique, sondage)
    elif systeme_vote == "STV / Ranked-Choice":
        return electeur.voter_stv(strategique, sondage)
    elif systeme_vote == "Approval Voting":
        return electeur.voter_approbation(strategique, sondage)
    
#=====================================================================================================
#================================GÉNÉRATION DE POP FIXE===============================================
#=====================================================================================================

def generer_population(nb_electeurs):
    population = []
    
    noms_factions = ["Pro-Bob", "Pro-Raoul", "Pro-Jackie", "Pro-Mark", "Pro-Francis"]
    probas_factions = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    for i in range(nb_electeurs):
        faction = np.random.choice(noms_factions, p=probas_factions)
        
        if faction == "Pro-Bob":
            score_bob     = np.random.normal(85, 10)
            score_raoul   = np.random.normal(45, 15)
            score_jackie  = np.random.normal(15, 10)
            score_mark    = np.random.normal(40, 20)
            score_francis = np.random.normal(10, 5)
            
        elif faction == "Pro-Raoul":
            score_bob     = np.random.normal(30, 15)    
            score_raoul   = np.random.normal(85, 10)  
            score_jackie  = np.random.normal(40, 15)
            score_mark    = np.random.normal(50, 15)
            score_francis = np.random.normal(20, 10)
            
        elif faction == "Pro-Jackie":
            score_bob     = np.random.normal(15, 10)    
            score_raoul   = np.random.normal(40, 15)  
            score_jackie  = np.random.normal(85, 10)
            score_mark    = np.random.normal(60, 15)
            score_francis = np.random.normal(30, 15)
            
        elif faction == "Pro-Mark":
            score_bob     = np.random.normal(40, 15)    
            score_raoul   = np.random.normal(50, 15)  
            score_jackie  = np.random.normal(60, 15)
            score_mark    = np.random.normal(85, 10)
            score_francis = np.random.normal(70, 15)
            
        else: # Pro-Francis
            score_bob     = np.random.normal(10, 10)    
            score_raoul   = np.random.normal(20, 15)  
            score_jackie  = np.random.normal(30, 15)
            score_mark    = np.random.normal(70, 15)
            score_francis = np.random.normal(85, 10)

        utilities = {
            "Bob": round(np.clip(score_bob, 0, 100), 1),
            "Raoul": round(np.clip(score_raoul, 0, 100), 1),
            "Jackie": round(np.clip(score_jackie, 0, 100), 1),
            "Mark": round(np.clip(score_mark, 0, 100), 1),
            "Francis": round(np.clip(score_francis, 0, 100), 1)
        }
        
        electeur = Electeur(id_electeur=i, utilities=utilities, faction=faction)
        population.append(electeur)
        
    return population


#=====================================================================================================
#===============================GENERATION DE POP ALÉATOIRE===========================================
#=====================================================================================================

def generer_election_aleatoire(nb_electeurs):
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
            utilities[nom] = round(np.clip(score, 0, 100), 1)
            
        electeur = Electeur(id_electeur=i, utilities=utilities, faction=nom_faction)
        population.append(electeur)
        
    return population, probabilites_factions, matrice_moyennes


#=====================================================================================================
#======================CALCUL SCORE DE BONHEUR DU MEUILLEUR CANDIDAT==================================
#=====================================================================================================

def calculer_optimum_social(population, liste_candidats):
    scores_totaux = {candidat: 0 for candidat in liste_candidats}

    for electeur in population:
        for candidat, score in electeur.utilities.items():
            scores_totaux[candidat] += score

    gagnant_optimal = max(scores_totaux, key=scores_totaux.get)
    score_max = scores_totaux[gagnant_optimal]
    return gagnant_optimal, score_max, scores_totaux


#=====================================================================================================
#==================================A UTILISER POUR LES TESTS==========================================
#=====================================================================================================

def fixer_aleatoire(seed_value=42):
    np.random.seed(seed_value)


#=====================================================================================================
#=============================================SONDAGE=================================================
#=====================================================================================================

def generer_sondage(population, liste_candidats, taille_echantillon=None):
    """
    Réalise un sondage d'intentions de vote (basé sur le 1er choix sincère).
    Si taille_echantillon est None, interroge toute la population.
    """
    if taille_echantillon is None or taille_echantillon >= len(population):
        sondes = population
    else:
        sondes = random.sample(population, taille_echantillon)
        
    intentions = {c: 0 for c in liste_candidats}
    for electeur in sondes:
        intentions[electeur.get_favori()] += 1
        
    total_sondes = len(sondes)
    sondage_pct = {c: round((v / total_sondes) * 100, 1) for c, v in intentions.items()}
    
    return sondage_pct


#=====================================================================================================
#==================GENERATION DE PROMPTS PAR GROUPE AYANT LES MEME PREFERENCES========================
#=====================================================================================================

def preparer_prompts_llm(echantillon_strategique, sondage_actuel, systeme_vote):
    """
    Regroupe les électeurs stratégiques par ordre de préférence exact 
    pour minimiser les appels API.
    """
    groupes_profils = {}
    
    for electeur in echantillon_strategique:
        profil_clé = tuple(electeur.get_classement())
        
        if profil_clé not in groupes_profils:
            groupes_profils[profil_clé] = {
                "exemple_electeur": electeur,
                "membres": []
            }
        groupes_profils[profil_clé]["membres"].append(electeur)
        
    prompts_a_envoyer = {}
    for profil_clé, data in groupes_profils.items():
        electeur_type = data["exemple_electeur"]
        
        prompt = f"""Tu es un citoyen votant de manière stratégique. 
Tes notes de préférence pour les candidats (sur 100) sont : {electeur_type.utilities}
Ton classement sincère est : {list(profil_clé)}

Le dernier sondage donne ces intentions de vote : {sondage_actuel}
Le mode de scrutin actuel est : {systeme_vote}.

Sachant que tu veux maximiser ton utilité finale et éviter l'élection des candidats que tu détestes, quel est ton vote stratégique ?
Réponds uniquement par le nom du candidat (ou la liste selon le scrutin), sans aucune autre phrase."""

        prompts_a_envoyer[profil_clé] = prompt
        
    return groupes_profils, prompts_a_envoyer


#=====================================================================================================
#===================================INTERROGATION DE GEMINI===========================================
#=====================================================================================================

def interroger_gemini(prompt, liste_candidats, max_tentatives=3):
    """
    Envoie le prompt à Gemini avec un système de réessai en cas de serveur surchargé.
    Nettoie la réponse pour s'assurer qu'il renvoie bien un candidat valide.
    """    
    for tentative in range(max_tentatives):
        try:
            reponse = client_gemini.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt
            )
            choix = reponse.text.strip()
            
            # Nettoyage
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
    

#=====================================================================================================
#===============================EXECUTION DU MODE LLM=================================================
#=====================================================================================================

def executer_mode_llm(groupes, prompts, liste_candidats):
    """
    Parcourt les profils uniques, interroge Gemini une seule fois par profil,
    et distribue la réponse à tous les électeurs de ce groupe.
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


#=====================================================================================================
#================================SIMULATION DE L'ÉLECTION COMPLÈTE====================================
#=====================================================================================================

def preparer_election_mixte(population, liste_candidats, pct_strategique, systeme_vote, taille_echantillon_sondage, mode_llm=False):
    """
    Sépare la population, génère le sondage, et applique les stratégies de vote.
    Retourne la liste complète des bulletins prêts à être comptés.
    """
    nb_strat = int(len(population) * (pct_strategique / 100))
    pop_melangee = population.copy()
    random.shuffle(pop_melangee)
    
    groupe_strategique = pop_melangee[:nb_strat]
    groupe_sincere = pop_melangee[nb_strat:]
    
    sondage = generer_sondage(population, liste_candidats, taille_echantillon=taille_echantillon_sondage)
    
    bulletins_finaux = []
    
    for electeur in groupe_sincere:
        bulletins_finaux.append(obtenir_vote_statique(electeur, systeme_vote, strategique=False, sondage=None))
        
    if not mode_llm:
        for electeur in groupe_strategique:
            bulletins_finaux.append(obtenir_vote_statique(electeur, systeme_vote, strategique=True, sondage=sondage))
 
    else:
        groupes, prompts = preparer_prompts_llm(groupe_strategique, sondage, systeme_vote)
        choix_par_profil = executer_mode_llm(groupes, prompts, liste_candidats)
        
        for profil_cle, data in groupes.items():
            vote_choisi = choix_par_profil[profil_cle]
            
            for electeur in data["membres"]:
                # Si le LLM a echoue, on repasse sur un vote sincere par securite
                if vote_choisi is None:
                    vote_choisi = obtenir_vote_statique(electeur, systeme_vote, strategique=False, sondage=None)
                
                # Formatage specifique si STV ou Approval : conversion de string en liste si necessaire
                if systeme_vote in ["STV / Ranked-Choice", "Approval Voting"] and isinstance(vote_choisi, str):
                    vote_choisi = [c.strip() for c in vote_choisi.split(',')]
                    
                bulletins_finaux.append(vote_choisi)
        
    return bulletins_finaux, sondage



#=====================================================================================================
#==========================================TEST=======================================================
#=====================================================================================================

if __name__ == "__main__":

    NB_TEST = 10000
    print(f"=== TEST SCÉNARIO FIXE ({NB_TEST} électeurs) ===")
    ma_pop_fixe = generer_population(NB_TEST)
    
    compteur_fixe = {f"Pro-{nom}": 0 for nom in CANDIDATS.values()}
    for e in ma_pop_fixe:
        compteur_fixe[e.faction] += 1
        
    for faction, compte in compteur_fixe.items():
        pourcentage = (compte / NB_TEST) * 100
        print(f"La faction {faction} représente {pourcentage:.1f}% de la population générée")    
    
    print(f"\n\n=== TEST SCÉNARIO ALÉATOIRE ({NB_TEST} électeurs) ===")
    ma_pop_alea, probas, _ = generer_election_aleatoire(NB_TEST)
    
    compteur_alea = {f"Pro-{nom}": 0 for nom in CANDIDATS.values()}
    for e in ma_pop_alea:
        compteur_alea[e.faction] += 1
        
    for faction, compte in compteur_alea.items():
        pourcentage = (compte / NB_TEST) * 100
        print(f"La faction {faction} représente {pourcentage:.1f}% de la population générée")


#=====================================================================================================
#=====================================================================================================
#=====================================================================================================