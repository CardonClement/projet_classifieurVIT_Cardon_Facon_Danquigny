Ce projet est un classifieur ViT se basant sur un set d'images de lettres de l'alphabet.
Avec 10 epoch, il a une précision finale de 76.66% (voir exemples dans Projet_MathisDanquigny_CardonClement_FaconNicolas_3APP.ipynb dans les output de cellules déjà existantes.)
Ce projet n'a qu'un commit, mais il a été fait par trois personnes depuis un seul poste : Cardon Clément, Danquigny Mathis et Facon Nicolas. 
Pour lancer l'execution, c'est simple : 

- Cloner ce projet 

- S'assurer d'avoir un gpu équipé de core cuda (nvidia), et d'avoir une version de torch compatible avec les core cuda. Si ce n'est pas le cas, il faut executer : 
pip uninstall torch (si torch déjà installé)
pip cache purge
pip install torch -f https://download.pytorch.org/whl/torch_stable.html

- Avoir une connexion à la première execution afin de télécharger le dataset

- Enfin, simplement lancer toutes les cellules une à une du notebook fourni (Projet_MathisDanquigny_CardonClement_FaconNicolas_3APP.ipynb).
  Les résultats attendus devraient tous s'y trouver.

Si vous ne voulez pas executer ce projet, il suffit de se référer aux outputs déjà présents dans le fichier Projet_MathisDanquigny_CardonClement_FaconNicolas_3APP.ipynb, résultat d'une execution au préalable en local avant de commit.

Pour toutes questions ou remarques, vous pouvez me contacter à l'adresse suivante : clementcardonpro@gmail.com
