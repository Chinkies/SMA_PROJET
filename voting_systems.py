import copy
import random


# =====================================================================================================
# ===================================== UTILITAIRES D'AUDIT ===========================================
# =====================================================================================================

def calculer_social_welfare(population, candidat):
    """Calcule le Bien-être Social total pour un candidat donné."""
    return sum(e.utilities[candidat] for e in population)


def departager_egalite(candidats):
    """Bris d'égalité déterministe par ordre alphabétique."""
    return sorted(candidats)[0]


# =============================================================================
# 1. PLURALITY / FIRST PAST THE POST
# =============================================================================

def vote_plurality(population, liste_candidats, taux_strategique=0.0, sondage=None):
    scores = {c: 0 for c in liste_candidats}
    nb_strat = int(len(population) * taux_strategique)
    indices_strat = set(random.sample(range(len(population)), nb_strat)) if nb_strat > 0 else set()

    for idx, electeur in enumerate(population):
        is_strat = idx in indices_strat
        bulletin = electeur.generer_bulletin_pluralite(strategique=is_strat, sondage=sondage)
        if bulletin in scores:
            scores[bulletin] += 1

    max_voix = max(scores.values())
    vainqueurs_potentiels = [c for c, v in scores.items() if v == max_voix]
    return departager_egalite(vainqueurs_potentiels), scores


# =============================================================================
# 2. TWO-ROUND RUNOFF VOTING
# =============================================================================

def vote_two_round(population, liste_candidats, taux_strategique=0.0, sondage=None):
    nb_electeurs = len(population)
    seuil_majorite = nb_electeurs / 2.0
    nb_strat = int(nb_electeurs * taux_strategique)
    indices_strat = set(random.sample(range(nb_electeurs), nb_strat)) if nb_strat > 0 else set()

    # --- TOUR 1 ---
    scores_t1 = {c: 0 for c in liste_candidats}
    for idx, electeur in enumerate(population):
        is_strat = idx in indices_strat
        bulletin = electeur.generer_bulletin_deux_tours(strategique=is_strat, sondage=sondage)
        scores_t1[bulletin] += 1

    candidats_tries_t1 = sorted(liste_candidats, key=lambda c: (scores_t1[c], -ord(c[0])), reverse=True)
    premier = candidats_tries_t1[0]

    if scores_t1[premier] > seuil_majorite:
        return premier, {"scores": scores_t1, "elu_au_tour_1": True}, None

    deuxieme = candidats_tries_t1[1]
    finalistes = [premier, deuxieme]

    # --- TOUR 2 ---
    scores_t2 = {premier: 0, deuxieme: 0}
    for electeur in population:
        u1 = electeur.utilities[premier]
        u2 = electeur.utilities[deuxieme]
        if u1 > u2:
            scores_t2[premier] += 1
        elif u2 > u1:
            scores_t2[deuxieme] += 1
        else:
            scores_t2[departager_egalite(finalistes)] += 1

    max_voix_t2 = max(scores_t2.values())
    vainqueur = departager_egalite([c for c, v in scores_t2.items() if v == max_voix_t2])
    return vainqueur, {"scores": scores_t1, "elu_au_tour_1": False, "qualifies": finalistes}, {"scores": scores_t2}


# =============================================================================
# 3. SINGLE TRANSFERABLE VOTE (IRV)
# =============================================================================

def vote_stv(population, liste_candidats, taux_strategique=0.0, sondage=None):
    nb_electeurs = len(population)
    seuil_majorite = nb_electeurs / 2.0
    nb_strat = int(nb_electeurs * taux_strategique)
    indices_strat = set(random.sample(range(nb_electeurs), nb_strat)) if nb_strat > 0 else set()

    bulletins = [
        electeur.generer_bulletin_stv(strategique=(i in indices_strat), sondage=sondage)
        for i, electeur in enumerate(population)
    ]
    candidats_actifs = set(liste_candidats)
    historique_etapes = []

    while True:
        scores_etape = {c: 0 for c in candidats_actifs}
        for bulletin in bulletins:
            for choix in bulletin:
                if choix in candidats_actifs:
                    scores_etape[choix] += 1
                    break

        historique_etapes.append(copy.deepcopy(scores_etape))
        max_voix = max(scores_etape.values())

        if max_voix > seuil_majorite or len(candidats_actifs) <= 2:
            candidats_max = [c for c, v in scores_etape.items() if v == max_voix]
            return departager_egalite(candidats_max), historique_etapes

        min_voix = min(scores_etape.values())
        a_eliminer = departager_egalite([c for c, v in scores_etape.items() if v == min_voix])
        candidats_actifs.remove(a_eliminer)


# =============================================================================
# 4. APPROVAL VOTING
# =============================================================================

def vote_approval(population, liste_candidats, taux_strategique=0.0, sondage=None, seuil_defaut=50):
    scores = {c: 0 for c in liste_candidats}
    nb_strat = int(len(population) * taux_strategique)
    indices_strat = set(random.sample(range(len(population)), nb_strat)) if nb_strat > 0 else set()

    for idx, electeur in enumerate(population):
        is_strat = idx in indices_strat
        bulletin = electeur.generer_bulletin_approbation(strategique=is_strat, sondage=sondage, seuil_defaut=seuil_defaut)
        if not bulletin:
            bulletin = [electeur.get_favori()]
        for candidat in bulletin:
            if candidat in scores:
                scores[candidat] += 1

    max_appr = max(scores.values())
    return departager_egalite([c for c, v in scores.items() if v == max_appr]), scores


# =============================================================================
# MOTEUR D'AUDIT GLOBAL COMPARATIF
# =============================================================================

def auditer_election(population, liste_candidats, taux_strategique=0.0, sondage=None, seuil_approval=50):
    """
    Exécute les 4 scrutins et compare les résultats à l'optimum social.
    """
    total_utilites = {c: calculer_social_welfare(population, c) for c in liste_candidats}
    optimum_candidat = max(total_utilites, key=total_utilites.get)
    optimum_sw = total_utilites[optimum_candidat]

    elu_fptp, _ = vote_plurality(population, liste_candidats, taux_strategique, sondage)
    elu_two_round, _, _ = vote_two_round(population, liste_candidats, taux_strategique, sondage)
    elu_stv, _ = vote_stv(population, liste_candidats, taux_strategique, sondage)
    elu_appr, _ = vote_approval(population, liste_candidats, taux_strategique, sondage, seuil_defaut=seuil_approval)

    scrutins = {
        "Plurality": elu_fptp,
        "Two-Round": elu_two_round,
        "STV": elu_stv,
        "Approval": elu_appr
    }

    audit = {}
    for nom, elu in scrutins.items():
        sw = total_utilites[elu]
        perte = optimum_sw - sw
        audit[nom] = {
            "vainqueur": elu,
            "social_welfare": round(sw, 1),
            "loss_absolue": round(perte, 1),
            "loss_pourcentage": round((perte / optimum_sw) * 100, 2) if optimum_sw > 0 else 0.0,
            "est_optimal": (elu == optimum_candidat)
        }

    return {"optimum": {"candidat": optimum_candidat, "sw": round(optimum_sw, 1)}, "resultats": audit}


#=====================================================================================================
#==========================================TEST=======================================================
#=====================================================================================================

if __name__ == "__main__":
    from population import generer_population, CANDIDATS, fixer_aleatoire, generer_sondage
    from voting_systems import auditer_election

    fixer_aleatoire(42)
    pop = generer_population(1000)
    candidats = list(CANDIDATS.values())
    sondage = generer_sondage(pop, candidats, taille_echantillon=200)

    print("--- SONDAGE INITIAL (Echantillon de 200) ---")
    for c, pct in sorted(sondage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {c:<10}: {pct}%")

    print("\n--- SCRUTIN 100% SINCERE ---")
    rapport_sincere = auditer_election(pop, candidats, taux_strategique=0.0, sondage=sondage)
    for nom, data in rapport_sincere["resultats"].items():
        print(f"  {nom:<15} -> Vainqueur: {data['vainqueur']:<10} | Perte: {data['loss_pourcentage']}%")

    print("\n--- SCRUTIN 50% STRATEGIQUE ---")
    rapport_mixte = auditer_election(pop, candidats, taux_strategique=0.5, sondage=sondage)
    for nom, data in rapport_mixte["resultats"].items():
        print(f"  {nom:<15} -> Vainqueur: {data['vainqueur']:<10} | Perte: {data['loss_pourcentage']}%")