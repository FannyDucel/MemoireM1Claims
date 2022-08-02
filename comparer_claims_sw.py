import pandas as pd
import json
import glob

def ouvrir_json(chemin):
    f = open(chemin, encoding="utf-8")
    toto = json.load(f)
    f.close()
    return toto

def ecrire_json(chemin, contenu):
    w = open(chemin, "w", encoding="utf-8")
    w.write(json.dumps(contenu, indent=2, ensure_ascii=False))
    w.close()

def lire_fichier(chemin):
    f = open(chemin, encoding="utf-8", errors="ignore")
    chaine = f.read()
    f.close()
    return chaine

def chemin_corpus(chemin_dossier, langue="en", taln_ou_acl="acl"):
    liste_chemins = []
    for chemin in glob.glob(chemin_dossier):
        # print(chemin)
        chaine = lire_fichier(chemin)
        # if detect(chaine)==langue:
        if taln_ou_acl == "acl":
            if "P" in chemin or "main" in chemin:
                if "the" in chaine:
                    liste_chemins.append(chemin)

    return liste_chemins


liste_chemins_acl = ouvrir_json("liste_chemins_acl.json")
df_clusters_intros = pd.read_csv("df_clusters_intros_stopwords.csv")  # v2
df_clusters_abstract = pd.read_csv("df_clusters_abstracts_stopwords.csv")
df_clusters_ccl = pd.read_csv("df_clusters_ccl_stopwords.csv")
df_clusters_corps = pd.read_csv("df_clusters_corps_stopwords.csv")

dico_total = {}
dico_clusters_abstract = {}
dico_clusters_intros = {}
dico_clusters_ccl = {}
dico_clusters_corps = {}

chemins_intros = ouvrir_json("claims_decoupes_intros_v06-2.json")
chemins_ccl = ouvrir_json("claims_decoupes_ccl_v06-2.json")
chemins_abstract = ouvrir_json("claims_decoupes_abstracts_v06-2.json")
chemins_corps = ouvrir_json("claims_decoupes_corps_v06-2.json")

#on réattribue chaque phrase et son cluster à l'article d'où elle est tirée
i = 0
for phrase in df_clusters_abstract["corpus"].dropna():
    cluster = df_clusters_abstract.loc[i]["cluster"]
    dico_clusters_abstract = ["abstract", cluster]
    i += 1
    for dico in chemins_abstract:
        for chemin, liste_phrases in dico.items():
            if phrase in liste_phrases:
                if chemin not in dico_total:
                    dico_total[chemin] = []
                dico_total[chemin].append(dico_clusters_abstract)

print("Boucle abstracts finie !")
# dico_clusters[df['cluster']

i = 0
for phrase in df_clusters_intros["corpus"].dropna():
    dico_clusters_intros = ["intro", df_clusters_intros.loc[i]["cluster"]]
    i += 1
    for dico in chemins_intros:
        for chemin, liste_phrases in dico.items():
            if phrase in liste_phrases:
                if chemin not in dico_total:
                    dico_total[chemin] = []
                dico_total[chemin].append(dico_clusters_intros)

print("Boucle intros finie !")

i = 0
for phrase in df_clusters_ccl["corpus"].dropna():
    dico_clusters_ccl = ["ccl", df_clusters_ccl.loc[i]["cluster"]]
    i += 1
    for dico in chemins_ccl:
        for chemin, liste_phrases in dico.items():
            if phrase in str(liste_phrases):
                if chemin not in dico_total:
                    dico_total[chemin] = []
                dico_total[chemin].append(dico_clusters_ccl)
print("Boucle ccl finie !")

i = 0
for phrase in df_clusters_corps["corpus"].dropna():
    dico_clusters_corps = ["corps", df_clusters_corps.loc[i]["cluster"]]
    i += 1
    for dico in chemins_corps:
        for chemin, liste_phrases in dico.items():
            if phrase in liste_phrases:
                if chemin not in dico_total:
                    dico_total[chemin] = []
                dico_total[chemin].append(dico_clusters_corps)
print("Boucle corps finie !")

ecrire_json("comparaison_claimsstopwordstotal.json", dico_total)
print("Fichier comparaison_claimsstopwordstotal écrit !")

# on rééquilibre après examen manuel afin que 0 = incertain, 1 = moyen, 2 = certain
for l in dico_total.values():
    for m in l:
        if m[0] == 'abstract':
            if m[1] == "one":
                m[1] = 2
            if m[1] == 'two':
                m[1] = 1
            if m[1] == 'zero':
                m[1] = 0
        if m[0] == 'ccl':
            if m[1] == "one":
                m[1] = 1
            if m[1] == "two":
                m[1] = 0
            if m[1] == "zero":
                m[1] = 2
        if m[0] == 'intro':
            if m[1] == "one":
                m[1] = 0
            if m[1] == "two":
                m[1] = 1
            if m[1] == "zero":
                m[1] = 2
        if m[0] == 'corps':
            if m[1] == "one":
                m[1] = 2
            if m[1] == "two":
                m[1] = 1
            if m[1] == "zero":
                m[1] = 0
ecrire_json("comparaison_claims_chiffres_equilibre_sw.json", dico_total)
print("Fichier comparaison_claims_chiffres_equilibre_sw écrit !")

comparaison_equilibree = ouvrir_json("comparaison_claims_chiffres_equilibre_sw.json")

dico_total = {}
# on ne garde qu'un claim max par partie, et celui avec le degré le + élevé
for chemin, lol in comparaison_equilibree.items():
    dico = {}
    for i in range(len(lol)):
        dico_total[chemin] = dico
        l = lol[i]
        partie = l[0]
        n_cluster = l[1]

        if partie not in dico_total[chemin]:
            dico[partie] = n_cluster
        else:
            if n_cluster > dico[partie]:
                dico[partie] = n_cluster
        dico_total[chemin] = dico
        # flat_ls = [item for sublist in ls for item in sublist]
print(dico_total)

ecrire_json("dico_comparaison_claims_sw.json", dico_total)
print("Fichier dico_comparaison_claims_sw écrit !")

dico_total = ouvrir_json("dico_comparaison_claims_sw.json")

# comparer les claims des différentes parties
dico_claims_diff = {}
dico_comparaison = ouvrir_json("dico_comparaison_claims_sw.json")
for article, sous_dico in dico_comparaison.items():
    # pas la peine de comparer si des claims détectés seulement dans une partie (par ex qu'en ccl)
    if len(sous_dico) > 1:
        # faire une liste des valeurs du dico
        values = list(sous_dico.values())
        # comparer avec la valeur précédente
        for i in range(1, len(values)):
            if values[i - 1] != values[i]:
                # print(article,sous_dico)
                dico_claims_diff[article] = sous_dico
print(dico_claims_diff)
ecrire_json("dico_claims_diff_sw.json", dico_claims_diff)

# comparer les claims des articles où on n'a qu'un degré
dico_claims_diff = {}
dico_comparaison = ouvrir_json("dico_comparaison_claims_sw.json")
for article, sous_dico in dico_comparaison.items():
    if len(sous_dico) == 1:
        dico_claims_diff[article] = sous_dico
print(dico_claims_diff)
ecrire_json("dico_claims_diff_bougentpas_sw.json", dico_claims_diff)

dico_claims_diff2 = {}
dico_comparaison = ouvrir_json("dico_comparaison_claims_sw.json")
for article, sous_dico in dico_comparaison.items():
    # pas la peine de comparer si des claims détectés seulement dans une partie (par ex qu'en ccl)
    if len(sous_dico) == 2:
        # faire une liste des valeurs du dico
        values = list(sous_dico.values())
        # comparer avec la valeur précédente
        for i in range(1, len(values)):
            if values[i - 1] != values[i]:
                # print(article,sous_dico)
                dico_claims_diff2[article] = sous_dico
print(len(dico_claims_diff2))
ecrire_json("dico_claims_diff2_sw.json", dico_claims_diff2)

dico_claims_diff3 = {}
dico_comparaison = ouvrir_json("dico_comparaison_claims_sw.json")
for article, sous_dico in dico_comparaison.items():
    # pas la peine de comparer si des claims détectés seulement dans une partie (par ex qu'en ccl)
    if len(sous_dico) == 3:
        # faire une liste des valeurs du dico
        values = list(sous_dico.values())
        # comparer avec la valeur précédente
        for i in range(1, len(values)):
            if values[i - 1] != values[i]:
                # print(article,sous_dico)
                dico_claims_diff3[article] = sous_dico
print(len(dico_claims_diff3))
ecrire_json("dico_claims_diff3_sw.json", dico_claims_diff3)


# comparer les claims des différentes parties
dico_claims_diff = {}
dico_comparaison = ouvrir_json("dico_comparaison_claims_sw.json")
for article, sous_dico in dico_comparaison.items():
    # pas la peine de comparer si des claims détectés seulement dans une partie (par ex qu'en ccl)
    # if len(sous_dico)>1:
    # faire une liste des valeurs du dico
    values = list(sous_dico.values())
    # comparer avec la valeur précédente
    for i in range(1, len(values)):
        if values[i - 1] != values[i]:
            # print(article,sous_dico)
            dico_claims_diff[article] = sous_dico
print(dico_claims_diff)
ecrire_json("dico_claims_diff_tt_sw.json", dico_claims_diff)

print(len(dico_comparaison), len(dico_claims_diff))

# trouver les parties qui ont les claims les + forts quand inégal dans un même article
dico_partiemaj = {}
# liste_sousdico=[]
for article, sous_dico in dico_claims_diff.items():
    liste_sousdico = []
    for partie, strength in sous_dico.items():
        # on crée une nouvelle liste (en gros on transforme les sous_dicos en liste)
        liste_sousdico.append(partie)
        liste_sousdico.append(strength)
        # au début on définit la valeur max comme la première clé du sousdico
        valeur_max = [liste_sousdico[0], liste_sousdico[1]]
        for i in range(2, len(liste_sousdico)):
            # puis on parcourt la liste, on prend seulement en compte les nombres (donc les valeurs impaires de la liste) et on les compare pour trouver le + grand et l'associer à sa partie (l'élément précédent dans la liste)
            if i % 2 != 0:
                if liste_sousdico[i] > valeur_max[1]:
                    valeur_max = [liste_sousdico[i - 1], liste_sousdico[i]]
        dico_partiemaj[article] = valeur_max

print(dico_partiemaj)

# trouver les parties qui ont les claims les + forts quand inégal dans un même article
dico_partiemaj = {}
# liste_sousdico=[]
for article, sous_dico in dico_total.items():
    liste_sousdico = []
    for partie, strength in sous_dico.items():
        # on crée une nouvelle liste (en gros on transforme les sous_dicos en liste)
        liste_sousdico.append(partie)
        liste_sousdico.append(strength)
        # au début on définit la valeur max comme la première clé du sousdico
        valeur_max = [liste_sousdico[0], liste_sousdico[1]]
        for i in range(2, len(liste_sousdico)):
            # puis on parcourt la liste, on prend seulement en compte les nombres (donc les valeurs impaires de la liste) et on les compare pour trouver le + grand et l'associer à sa partie (l'élément précédent dans la liste)
            if i % 2 != 0:
                if liste_sousdico[i] > valeur_max[1]:
                    valeur_max = [liste_sousdico[i - 1], liste_sousdico[i]]
        dico_partiemaj[article] = valeur_max

print(dico_partiemaj)

# trouver les parties qui ont les claims les + forts quand inégal dans un même article
dico_partiemaj = {}
# liste_sousdico=[]
for article, sous_dico in dico_claims_diff.items():
    liste_sousdico = []
    for partie, strength in sous_dico.items():
        # on crée une nouvelle liste (en gros on transforme les sous_dicos en liste)
        liste_sousdico.append(partie)
        liste_sousdico.append(strength)
        # au début on définit la valeur max comme la première clé du sousdico
        valeur_max = [liste_sousdico[0], liste_sousdico[1]]
        for i in range(2, len(liste_sousdico)):
            # puis on parcourt la liste, on prend seulement en compte les nombres (donc les valeurs impaires de la liste) et on les compare pour trouver le + grand et l'associer à sa partie (l'élément précédent dans la liste)
            if i % 2 != 0:
                if liste_sousdico[i] > valeur_max[1]:
                    valeur_max = [liste_sousdico[i - 1], liste_sousdico[i]]
        dico_partiemaj[article] = valeur_max

print(dico_partiemaj)
ecrire_json("dico_claims_partiemaj_sw.json", dico_partiemaj)


dico_compte_partiemaj = {"ccl": 0, "abstract": 0, "intro": 0, "corps": 0}
for article, liste in dico_partiemaj.items():
    if liste[0] == "ccl":
        dico_compte_partiemaj["ccl"] += 1
    if liste[0] == "abstract":
        dico_compte_partiemaj["abstract"] += 1
    if liste[0] == "intro":
        dico_compte_partiemaj["intro"] += 1
    if liste[0] == "corps":
        dico_compte_partiemaj["corps"] += 1
print(dico_compte_partiemaj)

ecrire_json("dico_claims_partiemaj_bougentpas_sw.json", dico_compte_partiemaj)

dico_claims_diff = ouvrir_json("dico_claims_diff_tt_sw.json")

# trouver les parties qui ont les claims les + forts, ou 3 dicos, 1 par partie type dico_abstract : {0:nb_claims_0,1:n, 2:n}
dico_abstract, dico_ccl, dico_intro, dico_corps = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0,
                                                                                                               1: 0,                                                                                                         2: 0}

for article, sous_dico in dico_comparaison.items():
    for partie, n_cluster in sous_dico.items():
        if partie == "abstract":
            dico_abstract[n_cluster] += 1
        if partie == "intro":
            dico_intro[n_cluster] += 1
        if partie == "ccl":
            dico_ccl[n_cluster] += 1
        if partie == "corps":
            dico_corps[n_cluster] += 1

dico_compteur = {"abstract": dico_abstract, "ccl": dico_ccl, "intro": dico_intro, "corps": dico_corps}
print(dico_compteur)

dico_nb_strength = {"0": dico_abstract[0] + dico_ccl[0] + dico_intro[0] + dico_corps[0],
                    "1": dico_abstract[1] + dico_ccl[1] + dico_intro[1] + dico_corps[1],
                    "2": dico_abstract[2] + dico_ccl[2] + dico_intro[2] + dico_corps[2]}
print(dico_nb_strength)
ecrire_json("dico_nb_strength_sw.json", dico_nb_strength)

# seulement sur les articles avec claims de force diff
dico_abstract, dico_ccl, dico_intro, dico_corps = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0,
                                                                                                               1: 0,
                                                                                                               2: 0}
for article, sous_dico in dico_claims_diff.items():
    for partie, n_cluster in sous_dico.items():
        if partie == "abstract":
            dico_abstract[n_cluster] += 1
        if partie == "intro":
            dico_intro[n_cluster] += 1
        if partie == "ccl":
            dico_ccl[n_cluster] += 1
        if partie == "corps":
            dico_corps[n_cluster] += 1

dico_compteur = {"abstract": dico_abstract, "ccl": dico_ccl, "intro": dico_intro, "corps": dico_corps}
print(dico_compteur)

dico_total_partie = {"abstract": sum(dico_compteur["abstract"].values()), 'ccl': sum(dico_compteur["ccl"].values()),
                     'intro': sum(dico_compteur["intro"].values()), 'corps': sum(dico_compteur["corps"].values())}
print(dico_total_partie)

# seulement sur les articles avec claims de même force
dico_abstract, dico_ccl, dico_intro, dico_corps = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0,
                                                                                                               1: 0,
                                                                                                               2: 0}

for article, sous_dico in dico_claims_diff.items():
    for partie, n_cluster in sous_dico.items():
        if partie == "abstract":
            dico_abstract[n_cluster] += 1
        if partie == "intro":
            dico_intro[n_cluster] += 1
        if partie == "ccl":
            dico_ccl[n_cluster] += 1
        if partie == "corps":
            dico_corps[n_cluster] += 1

dico_compteur = {"abstract": dico_abstract, "ccl": dico_ccl, "intro": dico_intro, "corps": dico_corps}
print(dico_compteur)

dico_total_partie = {"abstract": sum(dico_compteur["abstract"].values()), 'ccl': sum(dico_compteur["ccl"].values()),
                     'intro': sum(dico_compteur["intro"].values()), 'corps': sum(dico_compteur["corps"].values())}
print(dico_total_partie)

ecrire_json("dico_compteur_art_sw.json", dico_compteur)
ecrire_json("dico_total_partie_art_sw.json", dico_total_partie)

ecrire_json("dico_compte_partiemaj_sw.json", dico_compte_partiemaj)
ecrire_json("dico_compteur_sw.json", dico_compteur)
ecrire_json("dico_total_partie_sw.json", dico_total_partie)

dico_pays_acl = ouvrir_json("dico_pays_aclv06-2.json")
liste_pays = []
for chemin, liste in dico_pays_acl.items():
    for pays in liste:
        print(pays)
        if pays not in liste_pays:
            liste_pays.append(pays)
print(liste_pays)
europe = ["Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia", "Herzegovina",
          "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany",
          "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
          "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
          "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
          "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "UK", "Vatican City"]
europe = [e.lower() for e in europe]
america = ["usa", "us", "ecuador", "canada", "colombia", "brazil", "peru", "jersey", "mexico", "panama", "argentina",
           "chile", "uruguay", "cuba"]
asia = ["china", "macao", "mongolia", "singapore", "japan", "pakistan", "nepal", "jordan", "india", "qatar", "israel",
        "philippines", "indonesia", "thailand", "fiji", "bangladesh", "malaysia", "iraq"]
oceania = ["australia"]
africa = ["egypt", "morocco", "nigeria", "mali", "tunisia"]

# mixer avec les pays pour compter par pays/continent ?
dico_comparaison = ouvrir_json("dico_comparaison_claims_sw.json")
dico_abstract_eu, dico_ccl_eu, dico_intro_eu, dico_corps_eu = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0,
                                                                                                       2: 0}, {0: 0,
                                                                                                               1: 0,
                                                                                                               2: 0}
dico_abstract_am, dico_ccl_am, dico_intro_am, dico_corps_am = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0,
                                                                                                       2: 0}, {0: 0,
                                                                                                               1: 0,
                                                                                                               2: 0}
dico_abstract_as, dico_ccl_as, dico_intro_as, dico_corps_as = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0,
                                                                                                       2: 0}, {0: 0,
                                                                                                               1: 0,
                                                                                                               2: 0}
dico_abstract_oc, dico_ccl_oc, dico_intro_oc, dico_corps_oc = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0,
                                                                                                       2: 0}, {0: 0,
                                                                                                               1: 0,
                                                                                                               2: 0}
dico_abstract_af, dico_ccl_af, dico_intro_af, dico_corps_af = {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 0,
                                                                                                       2: 0}, {0: 0,
                                                                                                               1: 0,
                                                                                                               2: 0}

for article, sous_dico in dico_comparaison.items():
    if article in dico_pays_acl:

        for partie, n_cluster in sous_dico.items():
            # -> if article in liste_pays blablabla (ou faire par continent comme ça que 4 listes c'est moins relou)
            #         if article in dico_pays_acl:
            for pays in dico_pays_acl[article]:
                if pays.lower() in europe:
                    # print(partie,n_cluster)
                    if partie == "abstract":
                        dico_abstract_eu[n_cluster] += 1
                    if partie == "intro":
                        dico_intro_eu[n_cluster] += 1
                    if partie == "ccl":
                        dico_ccl_eu[n_cluster] += 1
                    if partie == "corps":
                        dico_corps_eu[n_cluster] += 1

                if pays.lower() in america:
                    if partie == "abstract":
                        dico_abstract_am[n_cluster] += 1
                    if partie == "intro":
                        dico_intro_am[n_cluster] += 1
                    if partie == "ccl":
                        dico_ccl_am[n_cluster] += 1
                    if partie == "corps":
                        dico_corps_am[n_cluster] += 1

                if pays.lower() in asia:
                    if partie == "abstract":
                        dico_abstract_as[n_cluster] += 1
                    if partie == "intro":
                        dico_intro_as[n_cluster] += 1
                    if partie == "ccl":
                        dico_ccl_as[n_cluster] += 1
                    if partie == "corps":
                        dico_corps_as[n_cluster] += 1

                if pays.lower() in oceania:
                    if partie == "abstract":
                        dico_abstract_oc[n_cluster] += 1
                    if partie == "intro":
                        dico_intro_oc[n_cluster] += 1
                    if partie == "ccl":
                        dico_ccl_oc[n_cluster] += 1
                    if partie == "corps":
                        dico_corps_oc[n_cluster] += 1

                if pays.lower() in africa:
                    if partie == "abstract":
                        dico_abstract_af[n_cluster] += 1
                    if partie == "intro":
                        dico_intro_af[n_cluster] += 1
                    if partie == "ccl":
                        dico_ccl_af[n_cluster] += 1
                    if partie == "corps":
                        dico_corps_af[n_cluster] += 1

dico_compteur_eu = {"abstract": dico_abstract_eu, "ccl": dico_ccl_eu, "intro": dico_intro_eu, "corps": dico_corps_eu}
dico_compteur_oc = {"abstract": dico_abstract_oc, "ccl": dico_ccl_oc, "intro": dico_intro_oc, "corps": dico_corps_oc}
dico_compteur_as = {"abstract": dico_abstract_as, "ccl": dico_ccl_as, "intro": dico_intro_as, "corps": dico_corps_as}
dico_compteur_af = {"abstract": dico_abstract_af, "ccl": dico_ccl_af, "intro": dico_intro_af, "corps": dico_corps_af}
dico_compteur_am = {"abstract": dico_abstract_am, "ccl": dico_ccl_am, "intro": dico_intro_am, "corps": dico_corps_am}

# compter le nombre d'articles qui ont une évolution dans tel ordre (x qui font 0-1-2, y 1-1-1, ...)
dico_claims_3 = ouvrir_json("dico_claims_diff3_sw.json")
# print(dico_claims_3)
dico_ordre = {}
for chemin, dico in dico_claims_3.items():
    ##ordre = tuple(dico.values()) #peut pas utiliser list comme clef
    ordre = str(dico.values())
    if ordre not in dico_ordre.keys():
        dico_ordre[ordre] = 0
    dico_ordre[ordre] += 1
print(dico_ordre)
# ecrire_json("dico_ordre3v06.json",dico_ordre)


ecrire_json("dico_ordre3_sw.json", dico_ordre)


dico_ordre = sorted(dico_ordre.items(), key=lambda x: x[1], reverse=True)
dico_ordre = dict(dico_ordre)
print(dico_ordre)

# faire les pourcentages de claims qui commencent par 0, par 1 etc

dico_pct_a = {0: 0, 1: 0, 2: 0}
dico_pct_i = {0: 0, 1: 0, 2: 0}
dico_pct_c = {0: 0, 1: 0, 2: 0}

for t, nb_occu in dico_ordre.items():
    # on compte le nombre de 1ers éléments (=abstract) qui correspondent à telle force de claim
    if int(t[13 + 0]) == 0:
        dico_pct_a[0] += nb_occu
    if int(t[13 + 0]) == 1:
        dico_pct_a[1] += nb_occu
    if int(t[13 + 0]) == 2:
        dico_pct_a[2] += nb_occu

    # idem pour les intros
    if int(t[15 + 1]) == 0:
        dico_pct_i[0] += nb_occu
    if int(t[15 + 1]) == 1:
        dico_pct_i[1] += nb_occu
    if int(t[15 + 1]) == 2:
        dico_pct_i[2] += nb_occu

    if int(t[17 + 2]) == 0:
        dico_pct_c[0] += nb_occu
    if int(t[17 + 2]) == 1:
        dico_pct_c[1] += nb_occu
    if int(t[17 + 2]) == 2:
        dico_pct_c[2] += nb_occu

dico_pct_total = {"début": dico_pct_a, "corps": dico_pct_i, "ccl": dico_pct_c}
print(dico_pct_total)
ecrire_json("dico_evol_strength_sw.json", dico_pct_total)