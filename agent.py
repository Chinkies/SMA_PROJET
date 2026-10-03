"""
agent.py - Définition de l'agent Électeur et de ses stratégies d'émission de bulletins.
"""

# =============================================================================
# CANDIDATS PAR DÉFAUT 
# =============================================================================
CANDIDATS = {0: "Bob", 1: "Raoul", 2: "Jackie", 3: "Mark", 4: "Francis"}


# =============================================================================
# CLASSE ÉLECTEUR
# =============================================================================
class Electeur:
    def __init__(self, id_electeur, utilities, faction="Inconnue", position=None):
        """
        Initialise un agent électeur.
        :param id_electeur: Identifiant unique (int)
        :param utilities: Dictionnaire {nom_candidat: score_utilite}
        :param faction: Libellé de faction ou nom de cluster d'origine
        :param position: Coordonnées 2D optionnelles (x, y) pour le modèle spatial
        """
        self.id = id_electeur
        self.utilities = utilities
        self.faction = faction
        self.position = position

    def get_favori(self):
        """Renvoie le nom du candidat ayant la plus haute utilité."""
        return max(self.utilities, key=self.utilities.get)

    def get_classement(self):
        """Renvoie la liste ordonnée des candidats par préférence décroissante."""
        return sorted(self.utilities, key=self.utilities.get, reverse=True)

    def get_approbations(self, seuil=50):
        """Renvoie la liste des candidats dont l'utilité atteint ou dépasse le seuil."""
        approbations = [candidat for candidat, score in self.utilities.items() if score >= seuil]
        if not approbations:
            approbations = [self.get_favori()]
        return approbations

    # -------------------------------------------------------------------------
    # STRATÉGIES D'ÉMISSION DE BULLETIN PAR MODE DE SCRUTIN
    # -------------------------------------------------------------------------

    def generer_bulletin_pluralite(self, strategique=False, sondage=None):
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
        return cand_a if self.utilities.get(cand_a, 0) >= self.utilities.get(cand_b, 0) else cand_b

    def generer_bulletin_deux_tours(self, strategique=False, sondage=None):
        """
        Two-Round Runoff Voting (1er tour)
        - Sincère : vote pour le favori.
        - Stratégique : sécurise sa voix en votant pour le meilleur candidat
          parmi les 3 plus grands prétendants au second tour.
        """
        favori = self.get_favori()
        if not strategique or not sondage:
            return favori

        viables = sorted(sondage, key=sondage.get, reverse=True)[:3]

        if favori in viables:
            return favori

        return max(viables, key=lambda c: self.utilities.get(c, 0))

    def generer_bulletin_stv(self, strategique=False, sondage=None):
        """
        Single Transferable Vote (STV)
        - Sincère : ordre de préférence réel décroissant.
        - Stratégique (Burying conditionnel) :
          Si le favori de l'électeur fait partie des deux favoris du sondage,
          l'électeur relègue son concurrent direct en dernière position pour
          éviter que des reports de voix ne le fassent passer devant.
          Sinon, il vote sincèrement pour maximiser l'efficacité de ses reports de voix.
        """
        classement_sincere = self.get_classement()
        if not strategique or not sondage:
            return classement_sincere

        favori = classement_sincere[0]
        leaders = sorted(sondage, key=sondage.get, reverse=True)[:2]

        if favori in leaders:
            rival = leaders[1] if leaders[0] == favori else leaders[0]
            bulletin = [c for c in classement_sincere if c != rival]
            bulletin.append(rival)
            return bulletin

        return classement_sincere

    def generer_bulletin_approbation(self, strategique=False, sondage=None, seuil_defaut=50):
        """
        Approval Voting
        - Sincère : approuve tout candidat dont l'utilité >= seuil_defaut.
        - Stratégique (Leader Rule / Duel au sommet) :
          Repère les 2 favoris du sondage (L1 et L2).
          L'électeur approuve son leader préféré parmi les deux et tous les candidats
          strictement meilleurs que ce leader, en excluant le rival de tête.
        """
        if not strategique or not sondage:
            return self.get_approbations(seuil=seuil_defaut)

        leaders = sorted(sondage, key=sondage.get, reverse=True)[:2]
        cand_a, cand_b = leaders[0], leaders[1]

        if self.utilities.get(cand_a, 0) >= self.utilities.get(cand_b, 0):
            leader_prefere = cand_a
            rival_craint = cand_b
        else:
            leader_prefere = cand_b
            rival_craint = cand_a

        seuil_strategique = self.utilities.get(leader_prefere, 0)

        approbations = [
            c for c, u in self.utilities.items()
            if u >= seuil_strategique and c != rival_craint
        ]

        if not approbations:
            approbations = [leader_prefere]

        return approbations


# =============================================================================
# AIGUILLAGE DU BULLETIN STATIQUE
# =============================================================================
def generer_bulletin_statique(electeur, systeme_vote, strategique=False, sondage=None, seuil_appr=50):
    """
    Aiguille l'électeur vers la bonne méthode d'émission selon le nom du scrutin.
    """
    if "Plurality" in systeme_vote:
        return electeur.generer_bulletin_pluralite(strategique=strategique, sondage=sondage)
    elif "Two-Round" in systeme_vote:
        return electeur.generer_bulletin_deux_tours(strategique=strategique, sondage=sondage)
    elif "STV" in systeme_vote:
        return electeur.generer_bulletin_stv(strategique=strategique, sondage=sondage)
    elif "Approval" in systeme_vote:
        return electeur.generer_bulletin_approbation(strategique=strategique, sondage=sondage, seuil_defaut=seuil_appr)
    else:
        # Repli par défaut sur le favori sincère
        return electeur.get_favori()