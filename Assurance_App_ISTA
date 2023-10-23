import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error






def App():

    st.image("Fleury.jpg", use_column_width= True
        )

    # Utilisation de la fonction title avec la balise span et l'attribut style pour la couleur
    #st.title("<span style='color: red;'>Mon Titre en Rouge", unsafe_allow_html=True)

    st.title(" Machine Learning App pour la Prédiction de la Prime d'Assurance dans une Compagnie d'Assurance"
        )
    st.subheader("Auteur : Fleury Niyokwizera"
        )
   

    # Fonction d'importation des donnees
    @st.cache_data()
    def data_load(file):
        mydata = pd.read_csv(file
        )
        return mydata
    

        # Affichage des donnees importees
    file = st.file_uploader("Upload your dataset in csv format ", type=["csv"]
        )
    if file is not None:
        mydata =data_load(file
        )

        if st.sidebar.checkbox("Afficher les Données Brutes de la Compagnie d'Assurance:", False):
            st.subheader(" Jeu de donnees de la compagnie d'assurance ")
            st.write(mydata
        )

       

        # Affichage d'un echantillon eleatoire voulu
        st.markdown("***Dans le dataset, pour les variables*** : \n\n **Sexe:**\n\n *0= Femme, 1= Homme* \n\n **Risk Profile:**\n\n *2= Grand Risque, 3= Risque Moyen, 4= Petit Risque* \n\n **Type_Contrat:**\n\n *10= Contrat Spécial, 100= Contrat Ordinaire*"
                    )
        Sample_mydata = mydata.sample(50
        )
        st.write(Sample_mydata
        )


        # Fonction pour train et test/ split
        seed= 125

        # Feature
        x = mydata.drop("Prime", axis = 1
        )

        # Target
        y = mydata["Prime"]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y, 
            test_size= 0.4, 
            random_state= seed

        )
        st.write(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # Creation d'une fonction permettant d'evaluer le modele

        
        def evaluatemodele (model):
            train_preds = model.predict(x_train)
            test_preds = model.predict(x_test)
            rmse_train = mean_squared_error(y_train, train_preds, squared= False).round(3)
            rmse_test = mean_squared_error(y_test, test_preds, squared= False).round(3)
            return rmse_train, rmse_test
        
        
        def evaluatemodele1 (model):
            train_preds = model.predict(x_train)
            test_preds = model.predict(x_test)
            rmse_train = mean_absolute_error(y_train, train_preds).round(3)
            rmse_test = mean_absolute_error(y_test, test_preds).round(3)
            return rmse_train, rmse_test
        
        
        def evaluatemodele2 (model):
            train_preds = model.predict(x_train)
            test_preds = model.predict(x_test)
            rmse_train = r2_score(y_train, train_preds).round(3)
            rmse_test = r2_score(y_test, test_preds).round(3)
            return rmse_train, rmse_test
        
        
    

        # Choisir les types d'algorthmes Machine learning a utiliser

        type_Algo = st.sidebar.selectbox(
        "Choisir le modèle de Prediction :",
        ("linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Machine ")
        )

        ############## Pour l'algorthme Random Forest######################

        if type_Algo== "Random Forest":
            st.sidebar.subheader("Les hyperparamètres du modèle"
        )
            n_abres = st.sidebar.number_input(
            "Choisir le nombre d'abres differents a entrainer",
                min_value= 100,
                max_value= 10000,
                step= 10
            
        )
            profondeur_arbres = st.sidebar.number_input(
            "Choisir la profondeur maximale de chaque arbre",
                min_value= 2,
                max_value= 200,
                step= 1
        )
            bootstrap_arbres = st.sidebar.radio(
            "Echantillon boostrap lors de la Creation d'abres",
            (True, False)
        )
            hyper_criterion = st.sidebar.selectbox(
                "le Critere statistique utilise pour couper les feuilles",
                ["squared_error", "absolute_error", "poisson",  "friedman_mse"]
        )
            hyper_n_jobs = st.sidebar.number_input(
                "le nombre de coeurs de CPU que vous utilisez pour la construction des arbres",
                min_value= -1,
                max_value= 10
        )
            max_featuresh = st.sidebar.number_input(
                "le nombre maximum de variables qu'on tire aleatoirement pour chaque arbre",
                min_value= 1,
                max_value= 10000,
                step= 1
        )
            min_samples_splith = st.sidebar.number_input(
                "le nombre minimum d'observations qu'il faut dans une feuille avant separation. ce critere evite le sur-apprentissage",
                min_value= 2,
                max_value= 10,
                step= 2
        )
            verboseh = st.sidebar.number_input(
                "Parametre qui permet de surveuiller la construction des arbres",
                min_value= 0,
                max_value= 5,
                step= 1
        )
    
    
    
        # Initialisation du modele 
            model1= RandomForestRegressor(
                n_estimators= n_abres,
                
                max_depth= profondeur_arbres,
                bootstrap= bootstrap_arbres,
                random_state= seed,
                criterion= hyper_criterion,
                n_jobs= hyper_n_jobs,
                max_features= max_featuresh,
                min_samples_split= min_samples_splith,
                verbose= verboseh
            
        )
        # Entrainement du modele
            model1.fit(x_train, y_train)
    
        # Fonction pour prédire la prime d'assurance

            def predict_prime(Age, Income, Sexe, Risk_Profile, Type_Contrat):
                if Sexe == 'Homme':
                    Sexe = 0
                else :
                    Sexe = 1
                if Risk_Profile== "Grand Risque":
                    Risk_Profile = 2
                elif Risk_Profile== "Risque Moyenne":
                    Risk_Profile = 3
                else:
                    Risk_Profile = 4
                if Type_Contrat== "Contrat Spéciale":
                    Type_Contrat= 10
                else:
                    Type_Contrat=100
                y_pred = model1.predict([[Age, Income, Sexe, Risk_Profile, Type_Contrat]])
                
                return y_pred[0]
               
            # les Caracteristiques (features) de l'assure 
            Age= st.number_input("Age", min_value= 18, max_value= 120)
            Income= st.number_input("Income", min_value= 200000)
            Sexe = st.selectbox("Sexe",options= ['Homme', 'Femme'])
            Risk_Profile =st.selectbox("Risk_Profile", ["Grand Risque","Risque Moyen", "Petit Risque"])
            Type_Contrat =st.selectbox("Type_Contrat", ["Contrat Spéciale", "Contrat Ordinaire"]
        )


        # Predictions
            if st.button("Prédire"):
                st.subheader("La Prime Prédite selon le profile enregistré de l'Assuré")
                result = predict_prime(Age, Income, Sexe, Risk_Profile, Type_Contrat)
                st.success(f"la Prime d'Asurance Prédite est de : {result.round(3)}"
        )

        
        # Evaluation de la performance du modele

                st.subheader("Les métriques de la performance du modèle du Random Forest")

                st.write("1. Le Coéfficient de Détérmination (R^2) : ",evaluatemodele2(model1))

                st.write("2. L'Erreur Quadratique Moyenne (RMSE)  : ",evaluatemodele(model1))  
                
                st.write("3. L'Erreur Absolue Moyenne (MAE)  : ",evaluatemodele1(model1))
                 
                


        ######### La Regression Lineaire#############

        if type_Algo== "linear Regression":
            st.sidebar.subheader("Les hyperparamètres du modèle"
        )

            fit_intercept1 = st.sidebar.radio(
            "Choisir votre fit_intercept",
            (True, False)
        )

            X_cope1 = st.sidebar.radio(
            "Choisir votre x_corpe",
            (True, False)
        )

            n_jobs1 = st.sidebar.number_input(
            "Enter votre n_jobs",
                min_value= -1,
                max_value= 1
        )

            positive1 = st.sidebar.radio(
            "Forcer le coéfficient a etre positif",
                ( False, True)
        )

        # Initialisation du modele 
            model1= LinearRegression(
                fit_intercept= fit_intercept1,
                copy_X= X_cope1,
                n_jobs= n_jobs1,
                positive= positive1,
                

        )

        # Entrainement du modele
            model1.fit(x_train, y_train
        )
    
        # Fonction pour prédire la prime d'assurance
            def predict_prime(Age, Income, Sexe, Risk_Profile, Type_Contrat):
                if Sexe == 'Homme':
                    Sexe = 0
                else :
                    Sexe = 1
                if Risk_Profile== "Grand Risque":
                    Risk_Profile = 2
                elif Risk_Profile== "Risque Moyenne":
                    Risk_Profile = 3
                else:
                    Risk_Profile = 4
                if Type_Contrat== "Contrat Spéciale":
                    Type_Contrat= 10
                else:
                    Type_Contrat=100
                y_pred = model1.predict([[Age, Income, Sexe, Risk_Profile, Type_Contrat]])
                
                return y_pred[0]
               
            

    
        # les Caracteristiques (features) de l'assure 
            Age= st.number_input("Age", min_value= 18, max_value= 120)
            Income= st.number_input("Income", min_value= 200000)
            Sexe = st.selectbox("Sexe",options= ['Homme', 'Femme'])
            Risk_Profile =st.selectbox("Risk_Profile", ["Grand Risque","Risque Moyenne", "Petit Risque"])
            Type_Contrat =st.selectbox("Type_Contrat", ["Contrat Spéciale", "Contrat Ordinaire"]
        )


        # Predictions
            if st.button("Prédire"):
                st.subheader("La Prime Prédite selon le profile enregistré de l'Assuré")
                result = predict_prime(Age, Income, Sexe, Risk_Profile,Type_Contrat)
                st.success(f"la Prime d'Asurance Prédite est de : {result.round(3)}"
        )

        
        # Evaluation de la performance du modele

                st.subheader("Les métriques de la performance du modèle de la Régression linéaire")

                st.write("1. Le Coéfficient de Détérmination (R^2) : ",evaluatemodele2(model1))

                st.write("2. L'Erreur Quadratique Moyenne (RMSE)  : ",evaluatemodele(model1))  
                
                st.write("3. L'Erreur Absolue Moyenne (MAE)  : ",evaluatemodele1(model1))
                 


        ######### Support Vector Machine Vector #############

        if type_Algo== "Support Vector Machine ":
            st.sidebar.subheader("Les hyperparamètres du modèle "
        )

            hyper_C= st.sidebar.number_input(
            "Choisir la pénalité de régularisation (C) ",
                min_value= 1.0,
                max_value= 10.0,
        )
            hyper_Karnel= st.sidebar.selectbox(
            "Choisir votre Kernel",
            ['linear', 'rbf', 'precomputed', 'poly', 'sigmoid']
        )
            hyper_degree= st.sidebar.number_input(
           " Choisir votre degree karnel polinomial",
            min_value=1,
            max_value= 10
        )
            hyper_gamma= st.sidebar.selectbox(
            "Choisir le type gamma",
            ["scale", "auto"]
        )
            hyper_size_cache= st.sidebar.number_input(
            "Choisir votre cache_size",
                min_value= 200,
                #max_value= 500,
                step = 50
        )
            hyper_max= st.sidebar.number_input(
            "Choisir le nombre maximal d'Iterations",
                min_value=-1,
        )
            hyper_tol= st.sidebar.number_input(
            " Ajuster la tolérance (tol) qui définit le critère d'arrêt de l'algorithme",
                min_value=0.001,
        )
            hyper_epslon= st.sidebar.number_input(
            " Ajuster le paramètre de marge douce",
                min_value=0.1,
        )
            

        # Initialisation du modele 
            model1= SVR(
                C= hyper_C,
                kernel= hyper_Karnel,
                degree= hyper_degree,
                gamma= hyper_gamma,
                cache_size= hyper_size_cache,
                max_iter= hyper_max,
                tol = hyper_tol,
                epsilon = hyper_epslon

                
        )

        # Entrainement du modele
            model1.fit(x_train, y_train)
        
    
        # Fonction pour prédire la prime d'assurance
            def predict_prime(Age, Income, Sexe, Risk_Profile, Type_Contrat):
                if Sexe == 'Homme':
                    Sexe = 0
                else :
                    Sexe = 1
                if Risk_Profile== "Grand Risque":
                    Risk_Profile = 2
                elif Risk_Profile== "Risque Moyenne":
                    Risk_Profile = 3
                else:
                    Risk_Profile = 4
                if Type_Contrat== "Contrat Spéciale":
                    Type_Contrat= 10
                else:
                    Type_Contrat=100
                y_pred = model1.predict([[Age, Income, Sexe, Risk_Profile, Type_Contrat]])
                
                return y_pred[0]
               
            

    
        # les Caracteristiques (features) de l'assure 
            Age= st.number_input("Age", min_value= 18, max_value= 120)
            Income= st.number_input("Income", min_value= 200000)
            Sexe = st.selectbox("Sexe",options= ['Homme', 'Femme'])
            Risk_Profile =st.selectbox("Risk_Profile", ["Grand Risque","Risque Moyenne", "Petit Risque"])
            Type_Contrat =st.selectbox("Type_Contrat", ["Contrat Spéciale", "Contrat Ordinaire"]
        )


        # Predictions
            if st.button("Prédire"):
                st.subheader("La Prime Prédite selon le profile enregistré de l'Assuré")
                result = predict_prime(Age, Income, Sexe, Risk_Profile,Type_Contrat)
                st.success(f"la Prime d'Asurance Prédite est de : {result.round(3)}"
        )

        
        # Evaluation de la performance du modele

                st.subheader("Les métriques de la performance du modèle du Support Vector Machine (SVM)")

                st.write("1. Le Coéfficient de Détérmination (R^2) : ",evaluatemodele2(model1))

                st.write("2. L'Erreur Quadratique Moyenne (RMSE)  : ",evaluatemodele(model1))  
                
                st.write("3. L'Erreur Absolue Moyenne (MAE)  : ",evaluatemodele1(model1))
                 


        ################ Decision Tree #############


        if type_Algo== "Decision Tree":
            st.sidebar.subheader("Les hyperparamètres du modèle"
        )
            hyper_criteria = st.sidebar.selectbox(
            "Choisir le Critere de mesure",
            ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                
            
        )
            profondeur_arbres = st.sidebar.number_input(
            "Choisir la profondeur maximale d'un arbre",
                min_value= 1,
                max_value= 100,
                step= 1
        )
            hyper_features = st.sidebar.number_input(
            "The number of features to consider when looking for the best split:",
            min_value=1,
            max_value=10000000,
            step = 1

            
        )
    
    
    
        # Initialisation du modele 
            model1= DecisionTreeRegressor(
                criterion=hyper_criteria,
                #max_depth= profondeur_arbres,
                max_features= hyper_features,
                random_state= seed
        )
        # Entrainement du modele
            model1.fit(x_train, y_train)
    
        # Fonction pour prédire la prime d'assurance
            def predict_prime(Age, Income, Sexe, Risk_Profile, Type_Contrat):
                if Sexe == 'Homme':
                    Sexe = 0
                else :
                    Sexe = 1
                if Risk_Profile== "Grand Risque":
                    Risk_Profile = 2
                elif Risk_Profile== "Risque Moyenne":
                    Risk_Profile = 3
                else:
                    Risk_Profile = 4
                if Type_Contrat== "Contrat Spéciale":
                    Type_Contrat= 10
                else:
                    Type_Contrat=100
                y_pred = model1.predict([[Age, Income, Sexe, Risk_Profile, Type_Contrat]])
                
                return y_pred[0]
               
            

    
        # les Caracteristiques (features) de l'assure 
            Age= st.number_input("Age", min_value= 18, max_value= 120)
            Income= st.number_input("Income", min_value= 200000)
            Sexe = st.selectbox("Sexe",options= ['Homme', 'Femme'])
            Risk_Profile =st.selectbox("Risk_Profile", ["Grand Risque","Risque Moyenne", "Petit Risque"])
            Type_Contrat =st.selectbox("Type_Contrat", ["Contrat Spéciale", "Contrat Ordinaire"]
        )


        # Predictions
            if st.button("Prédire"):
                st.subheader("La Prime Prédite selon le profile enregistré de l'Assuré")
                result = predict_prime(Age, Income, Sexe, Risk_Profile,Type_Contrat)
                st.success(f"la Prime d'Asurance Prédite est de : {result.round(3)}"
        )
        
        # Evaluation de la performance du modele

                st.subheader("Les métriques de la performance du modèle de l'Arbre de Décesion")

                st.write("1. Le Coéfficient de Détérmination (R^2) : ",evaluatemodele2(model1))

                st.write("2. L'Erreur Quadratique Moyenne (RMSE)  : ",evaluatemodele(model1))  
                
                st.write("3. L'Erreur Absolue Moyenne (MAE)  : ",evaluatemodele1(model1))
                 
        
        
        
        ############### Gradient Boosting ##################

        if type_Algo== "Gradient Boosting":
            st.sidebar.subheader("Les hyperparamètres du modèle"
        )
            hyper_criteria = st.sidebar.selectbox(
            "Choisir la fonction du cout utilise pour la descente du Gradient",
            ['squared_error', 'absolute_error', 'huber', 'quantile']
                
            
        )
            profondeur_arbres = st.sidebar.number_input(
            "Choisir la profondeur des arbres entraines",
                min_value= 8,#3
                max_value= 15,#10
                step= 1
        )
            hyper_features = st.sidebar.number_input(
            "The number of features to consider when looking for the best split:",
            min_value=1,
            max_value=10000000,
            step = 1

            
        )
            n_abres = st.sidebar.number_input(
            "Choisir le nombre d'estimateurs ou d'iterations ",
                min_value= 100,
                max_value= 10000,
                step= 10
            
        )
            min_samples_splith = st.sidebar.number_input(
                "le nombre minimum d'observations qu'il faut dans une feuille avant separation. ce critere evite le sur-apprentissage",
                min_value= 2,
                max_value= 10,
                step= 1
        )
            Learning_rateh = st.sidebar.number_input(
                "Definir le pas a chaque descente du gradient",
                min_value= 0.01,
                max_value= 0.1,
                

        )
            Subsampleh = st.sidebar.number_input(
                "Controle du nombre de tirage aleatoire",
                min_value= 0.5,
                max_value= 1.1,
                

        )
    
    
    
        # Initialisation du modele 
            model1= GradientBoostingRegressor(
                loss= hyper_criteria,
                n_estimators= n_abres,
                max_depth= profondeur_arbres,
                max_features= hyper_features,
                min_samples_split= min_samples_splith,
                learning_rate= Learning_rateh,
                subsample= Subsampleh,
                random_state= seed
        )
        # Entrainement du modele
            model1.fit(x_train, y_train)
    
        # Fonction pour prédire la prime d'assurance
            def predict_prime(Age, Income, Sexe, Risk_Profile, Type_Contrat):
                if Sexe == 'Homme':
                    Sexe = 0
                else :
                    Sexe = 1
                if Risk_Profile== "Grand Risque":
                    Risk_Profile = 2
                elif Risk_Profile== "Risque Moyenne":
                    Risk_Profile = 3
                else:
                    Risk_Profile = 4
                if Type_Contrat== "Contrat Spéciale":
                    Type_Contrat= 10
                else:
                    Type_Contrat=100
                y_pred = model1.predict([[Age, Income, Sexe, Risk_Profile, Type_Contrat]])
                
                return y_pred[0]
               
            

    
        # les Caracteristiques (features) de l'assure 
            Age= st.number_input("Age", min_value= 18, max_value= 120)
            Income= st.number_input("Income", min_value= 200000)
            Sexe = st.selectbox("Sexe",options= ['Homme', 'Femme'])
            Risk_Profile =st.selectbox("Risk_Profile", ["Grand Risque","Risque Moyenne", "Petit Risque"])
            Type_Contrat =st.selectbox("Type_Contrat", ["Contrat Spéciale", "Contrat Ordinaire"]
        )


        # Predictions
            if st.button("Prédire"):
                st.subheader("La Prime Prédite selon le profile enregistré de l'Assuré")
                result = predict_prime(Age, Income, Sexe, Risk_Profile,Type_Contrat)
                st.success(f"la Prime d'Asurance Prédite est de : {result.round(3)}"
        )

        
        # Evaluation de la performance du modele

                st.subheader("Les métriques de la performance du modèle du Gradient Boosting")

                st.write("1. Le Coéfficient de Détérmination (R^2) : ",evaluatemodele2(model1))

                st.write("2. L'Erreur Quadratique Moyenne (RMSE)  : ",evaluatemodele(model1))  
                
                st.write("3. L'Erreur Absolue Moyenne (MAE)  : ",evaluatemodele1(model1))
                 

if __name__=="__main__":
    App()
