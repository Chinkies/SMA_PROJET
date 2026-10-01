from collections import Counter
import copy


# =====================================================================================================
# ===================================== UTILITAIRES D'AUDIT ===========================================
# =====================================================================================================

def calculer_social_welfare(population, candidat):
    """Calcule le Bien-être Social total (somme des utilités) pour un candidat donné."""
    return sum(e.utilities[candidat] for e in population)


def departager_egalite(candidats):
    """Règle déterministe de bris d'égalité (ordre alphabétique pour la reproductibilité)."""
    return sorted(candidats)[0]


# =====================================================================================================
# ============================== 1. PLURALITY / FIRST PAST THE POST ===================================
# =====================================================================================================

def vote_plurality(population, liste_candidats):
    """
    Chaque électeur vote uniquement pour son candidat préféré (1er choix).
    Le candidat avec le plus de voix l'emporte.
    
    Retourne : (vainqueur, scores_detailles)
    """
    scores = {c: 0 for c in liste_candidats}
    
    for electeur in population:
        favori = electeur.get_favori()
        if favori in scores:
            scores[favori] += 1
            
    max_voix = max(scores.values())
    candidats_en_tete = [c for c, v in scores.items() if v == max_voix]
    vainqueur = departager_egalite(candidats_en_tete)
    
    return vainqueur, scores


# =====================================================================================================
# ================================= 2. TWO-ROUND RUNOFF VOTING ========================================
# =====================================================================================================

def vote_two_round(population, liste_candidats):
    """
    Tour 1 : Vote uninominal. Si un candidat a la majorité absolue (> 50%), il est élu.
    Sinon, les 2 premiers se qualifient pour le Tour 2.
    Tour 2 : Chaque électeur vote pour celui des 2 finalistes qu'il préfère.
    
    Retourne : (vainqueur, stats_tour1, stats_tour2)
    """
    nb_electeurs = len(population)
    seuil_majorite = nb_electeurs / 2.0
    
    # --- TOUR 1 ---
    scores_t1 = {c: 0 for c in liste_candidats}
    for electeur in population:
        scores_t1[electeur.get_favori()] += 1
        
    # Tri des candidats par voix décroissantes (avec bris d'égalité déterministe)
    candidats_tries_t1 = sorted(
        liste_candidats, 
        key=lambda c: (scores_t1[c], -ord(c[0])), 
        reverse=True
    )
    
    premier = candidats_tries_t1[0]
    
    # Victoire dès le 1er tour
    if scores_t1[premier] > seuil_majorite:
        return premier, {"scores": scores_t1, "elu_au_tour_1": True}, None
        
    deuxieme = candidats_tries_t1[1]
    finalistes = [premier, deuxieme]
    
    # --- TOUR 2 ---
    scores_t2 = {premier: 0, deuxieme: 0}
    for electeur in population:
        # L'électeur choisit le finaliste pour lequel son utilité est la plus forte
        u1 = electeur.utilities[premier]
        u2 = electeur.utilities[deuxieme]
        if u1 > u2:
            scores_t2[premier] += 1
        elif u2 > u1:
            scores_t2[deuxieme] += 1
        else:
            # Égalité parfaite d'utilité : bris déterministe
            scores_t2[departager_egalite(finalistes)] += 1
            
    max_voix_t2 = max(scores_t2.values())
    gagnants_t2 = [c for c, v in scores_t2.items() if v == max_voix_t2]
    vainqueur = departager_egalite(gagnants_t2)
    
    stats_t1 = {"scores": scores_t1, "elu_au_tour_1": False, "qualifies": finalistes}
    stats_t2 = {"scores": scores_t2}
    
    return vainqueur, stats_t1, stats_t2


# =====================================================================================================
# ======================= 3. SINGLE TRANSFERABLE VOTE / RANKED-CHOICE =================================
# =====================================================================================================

def vote_stv(population, liste_candidats):
    """
    Vote alternatif (Instant-Runoff Voting / STV à vainqueur unique) :
    1. On compte les 1ers choix actifs de chaque électeur.
    2. Si un candidat a > 50% des voix valides, il gagne.
    3. Sinon, le candidat avec le moins de 1ers choix est éliminé.
    4. Ses bulletins sont redistribués au prochain choix non éliminé.
    
    Retourne : (vainqueur, historique_etapes)
    """
    nb_electeurs = len(population)
    seuil_majorite = nb_electeurs / 2.0
    
    # On extrait le classement strict sincère de chaque électeur
    bulletins = [electeur.get_classement() for electeur in population]
    candidats_actifs = set(liste_candidats)
    historique_etapes = []
    
    while True:
        # Compte des 1ères préférences parmi les candidats encore en lice
        scores_etape = {c: 0 for c in candidats_actifs}
        for bulletin in bulletins:
            for choix in bulletin:
                if choix in candidats_actifs:
                    scores_etape[choix] += 1
                    break
                    
        historique_etapes.append(copy.deepcopy(scores_etape))
        
        # Condition 1 : Vérifier la majorité absolue
        max_voix = max(scores_etape.values())
        if max_voix > seuil_majorite:
            vainqueur = [c for c, v in scores_etape.items() if v == max_voix][0]
            return vainqueur, historique_etapes
            
        # Condition 2 : S'il ne reste que 2 candidats (ou 1)
        if len(candidats_actifs) <= 2:
            candidats_max = [c for c, v in scores_etape.items() if v == max_voix]
            vainqueur = departager_egalite(candidats_max)
            return vainqueur, historique_etapes
            
        # Élimination du dernier
        min_voix = min(scores_etape.values())
        candidats_derniers = [c for c, v in scores_etape.items() if v == min_voix]
        a_eliminer = departager_egalite(candidats_derniers)
        candidats_actifs.remove(a_eliminer)


# =====================================================================================================
# ===================================== 4. APPROVAL VOTING ============================================
# =====================================================================================================

def vote_approval(population, liste_candidats, seuil=50):
    """
    Vote par approbation : chaque électeur approuve tous les candidats
    dont l'utilité dépasse le seuil (au minimum son favori).
    Le candidat avec le plus d'approbations est élu.
    
    Retourne : (vainqueur, approbations_totales)
    """
    scores = {c: 0 for c in liste_candidats}
    
    for electeur in population:
        bulletin = electeur.get_approbations(seuil=seuil)
        # Sécurité : si la liste est vide, on prend son favori
        if not bulletin:
            bulletin = [electeur.get_favori()]
            
        for candidat in bulletin:
            if candidat in scores:
                scores[candidat] += 1
                
    max_appr = max(scores.values())
    candidats_en_tete = [c for c, v in scores.items() if v == max_appr]
    vainqueur = departager_egalite(candidats_en_tete)
    
    return vainqueur, scores


# =====================================================================================================
# ============================= MOTEUR D'AUDIT GLOBAL DE L'ÉLECTION ==================================
# =====================================================================================================

def auditer_election(population, liste_candidats, seuil_approval=50):
    """
    Exécute les 4 systèmes en mode sincère, compare avec l'optimum social,
    et calcule le Social Welfare ainsi que la perte d'utilité (Loss).
    """
    # 1. Calcul de l'optimum social théorique
    total_utilites = {c: calculer_social_welfare(population, c) for c in liste_candidats}
    optimum_candidat = max(total_utilites, key=total_utilites.get)
    optimum_sw = total_utilites[optimum_candidat]
    
    # 2. Exécution des 4 scrutins
    vainqueur_fptp, _ = vote_plurality(population, liste_candidats)
    vainqueur_two_round, _, _ = vote_two_round(population, liste_candidats)
    vainqueur_stv, _ = vote_stv(population, liste_candidats)
    vainqueur_approval, _ = vote_approval(population, liste_candidats, seuil=seuil_approval)
    
    resultats_scrutins = {
        "Plurality (FPTP)": vainqueur_fptp,
        "Two-Round Runoff": vainqueur_two_round,
        "STV / Ranked-Choice": vainqueur_stv,
        "Approval Voting": vainqueur_approval
    }
    
    # 3. Tableau récapitulatif
    audit = {}
    for systeme, elu in resultats_scrutins.items():
        sw_elu = total_utilites[elu]
        perte = optimum_sw - sw_elu
        pct_perte = (perte / optimum_sw) * 100 if optimum_sw > 0 else 0.0
        
        audit[systeme] = {
            "vainqueur": elu,
            "social_welfare": round(sw_elu, 2),
            "loss_absolue": round(perte, 2),
            "loss_pourcentage": round(pct_perte, 2),
            "est_optimal": (elu == optimum_candidat)
        }
        
    return {
        "optimum_theorique": {
            "candidat": optimum_candidat,
            "social_welfare": round(optimum_sw, 2)
        },
        "systemes": audit
    }


# =====================================================================================================
# ============================================== TEST =================================================
# =====================================================================================================

if __name__ == "__main__":
    from population import generer_population, CANDIDATS, fixer_aleatoire
    
    fixer_aleatoire(42)
    NB_VOTANTS = 1000
    candidats = list(CANDIDATS.values())
    
    print(f"--- GÉNÉRATION DE {NB_VOTANTS} ÉLECTEURS ---")
    pop = generer_population(NB_VOTANTS)
    
    rapport = auditer_election(pop, candidats, seuil_approval=50)
    
    opt = rapport["optimum_theorique"]
    print(f"\n[OPTIMUM SOCIAL THÉORIQUE] Gagnant : {opt['candidat']} (SW Total : {opt['social_welfare']})")
    print("-" * 75)
    print(f"{'Système':<22} | {'Élu':<10} | {'SW Obtenu':<12} | {'Loss (%)':<10} | {'Optimal ?'}")
    print("-" * 75)
    
    for sys_nom, data in rapport["systemes"].items():
        opt_str = "OUI" if data["est_optimal"] else "NON"
        print(f"{sys_nom:<22} | {data['vainqueur']:<10} | {data['social_welfare']:<12} | {data['loss_pourcentage']:<9}% | {opt_str}")
    print("-" * 75)