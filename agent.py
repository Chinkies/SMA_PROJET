"""
agent.py - Définition de l'agent Électeur et de ses stratégies d'émission de bulletins.
"""

import numpy as np

# =============================================================================
# CANDIDATS PAR DÉFAUT
# =============================================================================
CANDIDATS = {0: "Bob", 1: "Raoul", 2: "Jackie", 3: "Mark", 4: "Francis"}


# =============================================================================
# CLASSE ÉLECTEUR
# =============================================================================
class Electeur:
    def __init__(self, id_electeur, utilities, faction="Inconnue", position=None):
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

    def get_approbations(self, seuil=58):
        """
        En vote sincère par approbation, l'électeur approuve les candidats qu'il juge
        vraiment satisfaisants (au-dessus d'un seuil d'exigence et de sa moyenne personnelle).
        Cela permet d'approuver son favori ET un candidat de compromis proche (ex: Jackie).
        """
        moyenne_perso = float(np.mean(list(self.utilities.values())))
        seuil_effectif = max(float(seuil), moyenne_perso + 4.0)
        approbations = [
            candidat for candidat, score in self.utilities.items()
            if score >= seuil_effectif
        ]
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
        - Stratégique (Vote utile sur le Top 2 du sondage) :
          L'électeur regarde les 2 candidats en tête du sondage et vote pour celui
          qu'il préfère parmi ces deux-là.
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
        - Stratégique (Vote utile de qualification parmi le Top 3) :
          Sécurise la présence au second tour de son candidat préféré parmi les 3 premiers
          du sondage (évite le vote perdu sur les petits candidats).
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
        - Stratégique (Burying & Polarisation tactique) :
          L'électeur place son candidat viable préféré du Top 3 en tête, et enterre (relègue
          en fin de classement) à la fois le rival principal ET le candidat de compromis
          susceptible de battre son camp aux transferts de voix.
        """
        classement_sincere = self.get_classement()
        if not strategique or not sondage:
            return classement_sincere

        favori = classement_sincere[0]
        top_3 = sorted(sondage, key=sondage.get, reverse=True)[:3]
        leader_camp = favori if favori in top_3 else max(top_3, key=lambda c: self.utilities.get(c, 0))

        # Enterre son 2e choix sincère (souvent le candidat centriste de compromis) et le leader adverse
        deuxieme_sincere = classement_sincere[1]
        rival_principal = top_3[0] if top_3[0] != leader_camp else top_3[1]

        enterres = []
        if deuxieme_sincere != leader_camp:
            enterres.append(deuxieme_sincere)
        if rival_principal not in enterres and rival_principal != leader_camp:
            enterres.append(rival_principal)

        milieu = [c for c in classement_sincere if c != leader_camp and c not in enterres]
        return [leader_camp] + milieu + enterres

    def generer_bulletin_approbation(self, strategique=False, sondage=None, seuil_defaut=58):
        """
        Approval Voting
        - Sincère : approuve tous les candidats au-dessus de son seuil d'acceptabilité (favorise le consensus).
        - Stratégique (Bullet Voting / Troncature égoïste) :
          Pour éviter qu'un candidat de compromis (2e choix) ne batte son favori ou le leader
          de son camp, l'électeur stratégique pratique le Bullet Voting en n'approuvant qu'un
          seul candidat (son favori s'il est dans le Top 3, sinon son préféré du Top 2).
        """
        if not strategique or not sondage:
            return self.get_approbations(seuil=seuil_defaut)

        favori = self.get_favori()
        top_3 = sorted(sondage, key=sondage.get, reverse=True)[:3]

        if favori in top_3:
            # Bullet voting pur : retire l'approbation au candidat centriste de compromis !
            return [favori]

        top_2 = top_3[:2]
        choix_utile = max(top_2, key=lambda c: self.utilities.get(c, 0))
        return [favori, choix_utile]


# =============================================================================
# AIGUILLAGE DU BULLETIN STATIQUE
# =============================================================================
def generer_bulletin_statique(electeur, systeme_vote, strategique=False, sondage=None, seuil_appr=58):
    """Aiguille l'électeur vers la bonne méthode d'émission selon le nom du scrutin."""
    if "Plurality" in systeme_vote:
        return electeur.generer_bulletin_pluralite(strategique=strategique, sondage=sondage)
    elif "Two-Round" in systeme_vote:
        return electeur.generer_bulletin_deux_tours(strategique=strategique, sondage=sondage)
    elif "STV" in systeme_vote:
        return electeur.generer_bulletin_stv(strategique=strategique, sondage=sondage)
    elif "Approval" in systeme_vote:
        return electeur.generer_bulletin_approbation(strategique=strategique, sondage=sondage, seuil_defaut=seuil_appr)
    else:
        return electeur.get_favori()