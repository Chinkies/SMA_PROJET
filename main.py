import sys
from population import (
    CANDIDATS, 
    generer_population, 
    generer_election_aleatoire, 
    calculer_optimum_social, 
    preparer_election_mixte,
    fixer_aleatoire
)
# On importe les utilitaires de tes collègues, mais on recrée des versions allégées des systèmes de vote
# car on leur passe des listes de bulletins pré-calculés, et non plus des objets Electeur.
from voting_systems import departager_egalite

def main():
    print("=" * 60)
    print("   SIMULATEUR DE CHOIX SOCIAL & VOTE STRATÉGIQUE   ")
    print("=" * 60)
    
    # 1. PARAMÉTRAGE INTERACTIF
    try:
        nb_electeurs = int(input("\nNombre d'électeurs (ex: 1000) : ") or 1000)
        
        print("\nType de population :")
        print("1. Scénario fixe (Factions prédéfinies)")
        print("2. Scénario aléatoire complet")
        choix_pop = input("Choix (1 ou 2) [1] : ") or "1"
        
        pct_strat = float(input("\nPourcentage d'électeurs stratégiques (0 à 100) [10] : ") or 10.0)
        
        print("\nSystème de vote :")
        print("1. Plurality (FPTP)")
        print("2. Two-Round Runoff")
        print("3. STV / Ranked-Choice")
        print("4. Approval Voting")
        choix_sys = input("Choix (1 à 4) [1] : ") or "1"
        
        systemes_map = {
            "1": "Plurality (FPTP)",
            "2": "Two-Round Runoff",
            "3": "STV / Ranked-Choice",
            "4": "Approval Voting"
        }
        systeme_selectionne = systemes_map.get(choix_sys, "Plurality (FPTP)")
        
        taille_sondage = input("\nTaille de l'échantillon pour le sondage pré-électoral (ex: 200, ou Entrée pour 100%) : ")
        if taille_sondage.strip() == "":
            taille_sondage = None
        else:
            taille_sondage = int(taille_sondage)
            
        print("\nMode de réflexion pour les agents stratégiques :")
        print("1. Agents statiques (Algorithmes prédéfinis)")
        print("2. Agents LLM (Google Gemini - attention aux limites d'API)")
        choix_ia = input("Choix (1 ou 2) [1] : ") or "1"
        mode_llm_actif = (choix_ia == "2")
        
    except ValueError:
        print("\n[Erreur] Veuillez entrer des nombres valides. Annulation.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INITIALISATION DE L'ÉLECTION...")
    liste_candidats = list(CANDIDATS.values())
    
    # Fixer la seed pour avoir des résultats reproductibles pendant les tests
    fixer_aleatoire(42) 

    # 2. GÉNÉRATION DE LA POPULATION
    if choix_pop == "1":
        population = generer_population(nb_electeurs)
    else:
        population, _, _ = generer_election_aleatoire(nb_electeurs)
        
    optimum_candidat, optimum_sw, sw_details = calculer_optimum_social(population, liste_candidats)
    print(f"\n[ÉTAPE 1] Population de {nb_electeurs} électeurs générée.")
    print(f"L'Optimum Social Théorique (Candidat qui maximise le bonheur global) est : {optimum_candidat.upper()}")
    print(f"(Social Welfare Maximum : {round(optimum_sw, 1)})")
    
    # 3. L'ÉLECTION MIXTE (Le cœur de ta partie)
    print(f"\n[ÉTAPE 2] Préparation des votes ({systeme_selectionne})")
    print(f"- {100 - pct_strat}% de votes sincères")
    print(f"- {pct_strat}% de votes stratégiques (Mode {'LLM' if mode_llm_actif else 'Statique'})")
    
    bulletins_finaux, resultat_sondage = preparer_election_mixte(
        population, 
        liste_candidats, 
        pct_strat, 
        systeme_selectionne, 
        taille_sondage, 
        mode_llm=mode_llm_actif
    )
    
    print("\nSondage perçu par les agents stratégiques :")
    for cand, pct in resultat_sondage.items():
        print(f"  - {cand}: {pct}%")

    # 4. DÉPOUILLEMENT (On adapte les compteurs de tes collègues pour lire une simple liste de bulletins)
    print("\n[ÉTAPE 3] Dépouillement des bulletins...")
    scores_finaux = {c: 0 for c in liste_candidats}
    vainqueur_reel = None
    
    if systeme_selectionne == "Plurality (FPTP)":
        for vote in bulletins_finaux:
            if vote in scores_finaux:
                scores_finaux[vote] += 1
        vainqueur_reel = departager_egalite([c for c, v in scores_finaux.items() if v == max(scores_finaux.values())])
        
    elif systeme_selectionne == "Two-Round Runoff":
        # Pour faire simple dans le main, on compte juste le premier tour si l'IA renvoie une string
        # Une vraie implémentation à 2 tours demanderait que bulletins_finaux conserve les utilités
        for vote in bulletins_finaux:
             if isinstance(vote, str) and vote in scores_finaux:
                 scores_finaux[vote] += 1
        vainqueur_reel = departager_egalite([c for c, v in scores_finaux.items() if v == max(scores_finaux.values())])
        print("(Note: Pour le main interactif, le décompte 2-Tours est simplifié au T1)")

    elif systeme_selectionne == "Approval Voting":
        for bulletin_liste in bulletins_finaux:
            if isinstance(bulletin_liste, list):
                for choix in bulletin_liste:
                    if choix in scores_finaux:
                        scores_finaux[choix] += 1
        vainqueur_reel = departager_egalite([c for c, v in scores_finaux.items() if v == max(scores_finaux.values())])

    elif systeme_selectionne == "STV / Ranked-Choice":
         for bulletin_liste in bulletins_finaux:
            if isinstance(bulletin_liste, list) and len(bulletin_liste) > 0:
                choix_1 = bulletin_liste[0]
                if choix_1 in scores_finaux:
                     scores_finaux[choix_1] += 1
         vainqueur_reel = departager_egalite([c for c, v in scores_finaux.items() if v == max(scores_finaux.values())])
         print("(Note: Pour le main interactif, le décompte STV affiche le vainqueur aux 1ères préférences)")

    # 5. RÉSULTATS & AUDIT
    print("\n" + "=" * 60)
    print("                      RÉSULTATS                      ")
    print("=" * 60)
    
    print(f"Vainqueur de l'élection ({systeme_selectionne}) : {vainqueur_reel.upper()}\n")
    print("Scores bruts (1ère préférence ou approbations) :")
    for cand, score in sorted(scores_finaux.items(), key=lambda item: item[1], reverse=True):
         print(f"  - {cand}: {score} voix")
         
    # Calcul de la Utility Loss
    sw_vainqueur = sw_details[vainqueur_reel]
    perte_absolue = optimum_sw - sw_vainqueur
    perte_pct = (perte_absolue / optimum_sw) * 100 if optimum_sw > 0 else 0
    
    print("\n--- AUDIT DU BIEN-ÊTRE SOCIAL ---")
    if vainqueur_reel == optimum_candidat:
        print("✅ SUCCÈS : Le système a élu le candidat socialement optimal.")
        print(f"Utility Loss : 0% (Le bien-être maximum de {round(optimum_sw, 1)} a été atteint)")
    else:
        print("❌ ÉCHEC : Le vainqueur n'est PAS le candidat optimal pour la société.")
        print(f"Le système a élu {vainqueur_reel} au lieu de {optimum_candidat}.")
        print(f"Utility Loss : -{round(perte_absolue, 1)} points d'utilité ({round(perte_pct, 2)}% de perte de bonheur global).")
        
    print("\nSimulation terminée.")

if __name__ == "__main__":
    main()