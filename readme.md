# GIT CONTENANT LES FICHIERS DE CODE PYTHON PRINCIPAUX UTILISES POUR MON MEMOIRE DE M1 SUR LES CLAIMS

On suppose que les données textuelles sont dans le dossier "data" (ici vide, mais je peux fournir les données si besoin ; contactez-moi)

Les fichiers doivent être exécutés dans l'ordre suivant :

* I) **extraire_claims.ipynb** : pattern-matching pour extraire les phrases contenant des indices de claims, et ce dans chaque partie des articles (résumés, introductions, corps, conclusion)

* I') **stats_claims.ipynb** : pas essentiel mais permet d'obtenir des statistiques et des tendances sur les claims précédemment extraits

* IIa) **metriques_eval.py** : fait le clustering et donne les différents scores des métriques d'évaluation => pour tester différentes combinaisons de paramètres et garder la meilleure pour la suite

* IIb) **classif-avecstopwords.py** : fait le clustering, fournit les mots-clefs de chaque cluster et les graphiques de visualisation (à exécuter seulement avec la meilleure combinaison de paramètres trouvés ; refait le clustering mais en sauvegardant les données et donnant les mots-clefs et visualisations)

* III) **comparer_claims_sw.py** : rattache les claims clusterisés aux articles dont ils sont issus et fait en sorte que le degré 0 égale toujours 'incertain', 1 = moyennement certain, 2 = certain ; la 2e partie du code garde seulement 1 claim par partie max (celui qui a le degré le + fort) ; puis préparation des données sur les continents et des données pour voir l'évolution (en intro claim de niveau 1, en corps de niveau 2, en ccl de niveau 0, ...)

* IV) **extraire_institutions.ipynb** : pattern-matching sur les 1000ers caractères des articles pour mettre à part ceux qui sont affiliés aux institutions de la liste (GAFAM + top 50 universités du classement de Shanghai 2021)

* IV') **extraire_pays_continents.ipynb** : pattern-matching sur les 1000ers caractères des articles pour trouver les pays associés aux auteur-ices et les associer à leur continent ; également des cellules pour voir l'évolution diachronique du nombre d'articles associés à tel ou tel continent et pour voir combien d'articles sont écrits majoritairement par des hommes, des femmes ou une égalité selon les continents

* IV'') **extraire_prenoms_genres.ipynb** : extraction des prénoms d'auteur-ices grâce aux fichiers bib, puis association de ces prénoms à leur genre, puis 1er décompte selon le genre majoritaire et 2e décompte selon le genre du/de la 1er-e auteur-ice ; dernière cellule pour l'évolution diachronique du nombre d'autrices

* V) **generation_graphiques/** : chaque fichier permet de générer des graphiques à partir du clustering des claims, soit en prenant en compte tout le corpus (_total), soit les sous-corpus liés aux institutions, aux continents ou au genre.
