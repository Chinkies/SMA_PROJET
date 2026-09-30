import numpy as np

CANDIDATS = {0: "Bob", 1: "Raoul", 2: "jackie", 3: "Mark", 4: "Francis"}

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
        return [candidat for candidat, score in self.utilities.items() if score >= seuil]


def generer_population(nb_electeurs):
    population = []
    
    noms_factions = ["Pro-Bob", "Pro-Raoul", "Pro-jackie", "Pro-Mark", "Pro-Francis"]
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
            
        elif faction == "Pro-jackie":
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
            "jackie": round(np.clip(score_jackie, 0, 100), 1),
            "Mark": round(np.clip(score_mark, 0, 100), 1),
            "Francis": round(np.clip(score_francis, 0, 100), 1)
        }
        
        electeur = Electeur(id_electeur=i, utilities=utilities, faction=faction)
        population.append(electeur)
        
    return population


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
        