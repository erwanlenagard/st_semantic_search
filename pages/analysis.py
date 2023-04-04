from lib import semantic_search
from lib.semantic_search import *

def main():
    #on initialise nos variables
    if 'query' not in st.session_state:
        st.session_state['query'] = 'écologie'
    if 'similarity_threshold' not in st.session_state:
        st.session_state['similarity_threshold'] = 0.8
    if 'n_neighbors' not in st.session_state:
        st.session_state['n_neighbors'] = 5
    if 'min_size' not in st.session_state:
        st.session_state['min_size'] = 5
    if 'max_size' not in st.session_state:
        st.session_state['max_size'] = 20
    
    #on récupère le nom de projet
    projet=st.session_state['projet']
    st.write(st.session_state['projet'])
    
    #on charge le fichier de données correspondant
    if os.path.isfile(os.path.join('.',projet,'data.pkl')):     
        df=load_pickle(projet)
        
        #on initialise nos variables en fonction des data chargées
        cat_posts=sorted(list(df["categorie_engagement"].unique()))
        top_n=len(df)
        
        if 'engagement_threshold' not in st.session_state:
            st.session_state['engagement_threshold'] = int(df['total_engagement'].max()/10)
        if 'cat_posts' not in st.session_state:
            st.session_state['cat_posts'] = cat_posts
        if 'top_n' not in st.session_state:
            st.session_state['top_n'] = top_n
        
        ################################################################################
        # SIDEBAR
        ################################################################################
        #CHAMPS DE RECHERCHE
        query = st.sidebar.text_input("Recherche", value=st.session_state['query'])
        
        # SEUIL DE SIMILARITE
        similarity_threshold = st.sidebar.slider("Seuil de similarité", 
                                               min_value=0.0, 
                                               max_value=1.0, 
                                               value=st.session_state['similarity_threshold'],
                                               step=0.01)
        # SEUIL D'ENGAGEMENT
        engagement_threshold = st.sidebar.number_input("Seuil d'engagement", 
                                                     min_value=1,
                                                     max_value=int(df['total_engagement'].max()),
                                                     value=st.session_state['engagement_threshold'],
                                                     step=1)
        # CATEGORIES DE POSTS
        cat_post = st.sidebar.multiselect("Catégorie de posts",
                                        cat_posts, 
                                        default=st.session_state['cat_posts'])
        
        # TAILLE DES POINTS
        size = st.sidebar.slider("Taille des points",
                               min_value=1,
                               max_value=50,
                               value=(st.session_state['min_size'],st.session_state['max_size']),
                               step=1)
        
        # UMAP - NOMBRE DE VOISINS
        n_neighbors = st.sidebar.slider("UMAP",
                                        min_value=3,
                                        max_value=100,
                                        value=st.session_state['n_neighbors'],
                                        step=1)
        

              
        # On met à jour nos variables
        st.session_state['query'] =query
        st.session_state['similarity_threshold'] =similarity_threshold
        st.session_state['engagement_threshold'] =engagement_threshold
        st.session_state['cat_posts'] =cat_post
        st.session_state['n_neighbors'] =n_neighbors
        st.session_state['min_size'] =size[0]
        st.session_state['max_size'] =size[1]
        
        ######################################################################################################
        # TRAITEMENTS
        ######################################################################################################
        
        # Si on a une requête
        if st.session_state['query'] is not None:

            # ON FILTRE NOTRE DATASET 
            ssr = search_reviews(df[df['text'].str.len()>0], 
                                                   st.session_state['query'],
                                                   st.session_state['top_n'])
            
            ssr=ssr[ssr['similarity']>st.session_state['similarity_threshold']]
            ssr=ssr[ssr["categorie_engagement"].isin(st.session_state['cat_posts'])]
            
            # ON REDUIT LE NB DE DIMENSIONS DU VECTEUR OPENAI
            umap_embeddings = process_umap(list(ssr['embeddings']), 
                                           n_neighbors=st.session_state['n_neighbors'], 
                                           n_components=2, 
                                           min_dist=0.0, 
                                           metric='cosine')
            
            ssr['x']=umap_embeddings[:,0]
            ssr['y']=umap_embeddings[:,1]
            
            # ON CLUSTERISE
            clusterer,labels,probabilities = hdbscan_clustering(umap_embeddings, 
                                                   alpha=1.0, 
                                                   cluster_selection_epsilon=0.5,
                                                   min_cluster_size=3,
                                                   min_samples=1)
            
            ssr[['cluster','cluster_proba']] = list(zip(labels,probabilities))
            ssr['cluster'] = ssr['cluster'].astype(int).astype(str)
            
            ################################################################################################
            # PARAMETRES DES VIZ
            ################################################################################################
            
            # Palette de couleurs pour les catégories de réaction
            color_palette= { 
                "sous reaction": "#8F8F8F",
                "sur-comments": "#009d9a", 
                "sur-comments sur-reactions" : "#8a3ffc", 
                "sur-comments sur-shares": "#003f5c", 
                "sur-reactions" : "#ffa600",
                "sur-shares sur-reactions" : "#ff7c43", 
                "sur-shares" : "#33b1ff",
                "sur reaction totale": "#fa4d56" }

            # Palette de couleurs pour les catégories d'émotions
            color_palette_emotion= { 
                "no_reactions": "#8F8F8F",
                "positive_reactions": "#76b7b2", 
                "negative_reactions" : "#b07aa1", 
                "haha_reactions": "#edc948"
            }
            
            # Taille des points mis à l'échelle
            scaled_values=scale_values(ssr["total_engagement"], st.session_state['min_size'], st.session_state['max_size'])
            
            ###################################################################################################
            # CREATION DES VIZ 
            ###################################################################################################
            
            # BAR CHART - COUVERTURE & RESONANCE GLOBALE
            fig_couverture_globale=couverture_resonance(df)

            # BAR CHART - COUVERTURE DU SUJET
            df_cr = cr_comparison(df,ssr)
            fig_couverture_topic= barplot_comparison(df_cr, 
                               "date", 
                               "uniq_id_all", 
                               "uniq_id_subset", 
                               'silver', 
                               'steelblue', 
                               'Couverture du sujet', 
                               'Date', 
                               '#Couverture')
            
            # BAR CHART - RESONNANCE DU SUJET
            fig_resonance_topic= barplot_comparison(df_cr, 
                   "date", 
                   "total_engagement_all", 
                   "total_engagement_subset", 
                   'silver', 
                   'steelblue', 
                   'Résonnance du sujet', 
                   'Date', 
                   '#Résonnance')
            
            # SCATTER 3D - Modélisation de la réaction
            colors = [color_palette[k] for k in ssr["categorie_engagement"].values]
            fig_surreaction3D =scatter3D(ssr,
                          scaled_values,
                          colors,
                          "Modélisation de la réaction", 
                          "categorie_engagement",
                          "tx_shares",
                          "tx_reactions",
                          'tx_comments',
                          "Sur-partage",
                          "Sur-like",
                          "Sur-commentaire")
            
            # SCATTER 3D - Modélisation de la réaction avec les émotions
            colors_emotion = [color_palette_emotion[k] for k in ssr["top_reaction"].values]
            fig_emotion=scatter3D(ssr,
                          scaled_values,
                          colors_emotion,
                          "Modélisation de la réaction", 
                          "top_reaction",
                          "tx_shares",
                          "tx_reactions",
                          'tx_comments',
                          "Sur-partage",
                          "Sur-like",
                          "Sur-commentaire")
            
            # WORDCLOUD
            wordcloud_topic,top_n_words=generate_wordcloud(list(ssr['lemmas']), 50)
            word_freq_df = pd.DataFrame(list(top_n_words.items()), columns=['word', 'frequency'])
            
            # SCATTER PLOT - CLUSTERING
            fig_clustering = scatter_clustering(ssr, 
                                                "x", 
                                                "y", 
                                                ["author_name", "short_text", "comments", "shares", "reactions"], 
                                                'total_engagement', 
                                                "cluster", 
                                                "Clustering HDBSCAN")
            
            # TABLEAU DE DONNEES FINAL
            cols=['uniq_id',
                  'post_url',
                  'author_id',
                  'author_name',
                  'created_time', 
                  'text', 
                  'total_engagement', 
                  'reactions', 
                  'comments', 
                  'shares', 
                  'categorie_engagement',
                  'top_reaction',
                  'similarity'
                 ]
            displayed_df=ssr[cols]
            
            ###################################################################################################
            # UPLOAD DES VIZ 
            ###################################################################################################
            
            # PARAMETRES DE PYPLOT
            username = 'erwan_ln' # your username
            api_key = 'DXVsMt23NaWJwbuSb6c3' # your api key - go to profile > settings > regenerate key
            
            # Stocker la viz sur Plotly et générer un embed code
            embed_code = False

            if embed_code == True:
                URL1, EMBED1 = upload_viz(username, 
                                     api_key, 
                                     fig_couverture_globale,
                                     projet + " - Couverture et résonnance")
                URL2, EMBED2 = upload_viz(username, 
                                     api_key, 
                                     fig_couverture_topic, 
                                     projet + " - Couverture du sujet ("+st.session_state['query']+")")
                URL3, EMBED3 = upload_viz(username,
                                     api_key,
                                     fig_resonance_topic, 
                                     projet + " - Résonnance du sujet ("+st.session_state['query']+")")
                URL4, EMBED4 = upload_viz(username,
                                     api_key,
                                     fig_surreaction3D, 
                                     projet + " - Modélisation de la réaction")
                URL5, EMBED5 = upload_viz(username,
                                     api_key,
                                     fig_emotion, 
                                     projet + " - Modélisation de la réaction & émotions")
                URL6, EMBED6 = upload_viz(username,
                                     api_key,
                                     fig_clustering, 
                                     projet + " - Clustering UMAP")
                print("* URLs DES VIZ >> ", URL1, '\n', URL2, '\n', URL3, '\n',URL4,'\n',URL5,'\n',URL6)
                print("\n*CODE EMBED A COLLER \n",EMBED1,'\n', EMBED2, '\n', EMBED3, '\n',EMBED4,'\n',EMBED5,'\n',EMBED6)         
         

            ###############################################################################################
            # AFFICHAGE - CONTAINER #1
            ###############################################################################################

            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("✍️ #posts",value=len(ssr))
                with col2:
                    st.metric("#engagements",value=ssr['total_engagement'].sum())
                with col3:
                     st.metric("🗫 #comments",value=ssr['comments'].sum())
                with col4:
                     st.metric("#shares",value=ssr['shares'].sum())
                with col5:
                     st.metric("#reactions",value=ssr['reactions'].sum())
                with col6:
                    st.metric("#AVG_engagements",value=round(ssr['total_engagement'].mean(),1))
                    
            ###############################################################################################
            # AFFICHAGE - CONTAINER #2
            ###############################################################################################

            with st.container():
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Couverture & résonance", 
                                                              "📈 Sur-réaction",
                                                              "❤️ Emotion",
                                                              ":hash: Wordcloud" , 
                                                              "Topics", 
                                                              "✍️ Data"])
                
                ###########################################################################################
                # ONGLET 1 - 📊 Couverture & résonance
                ###########################################################################################
                tab1.plotly_chart(fig_couverture_globale,use_container_width=True)
                tab1.plotly_chart(fig_couverture_topic,use_container_width=True)
                tab1.plotly_chart(fig_resonance_topic,use_container_width=True)
                
                ###########################################################################################
                # ONGLET 2 - 📈 Sur-réaction
                ###########################################################################################
                tab2.plotly_chart(fig_surreaction3D,use_container_width=True)
                
                ###########################################################################################
                # ONGLET 3 - ❤️ Emotion
                ###########################################################################################
                tab3.plotly_chart(fig_emotion,use_container_width=True)
                
                ###########################################################################################
                # ONGLET 4 - :hash: Wordcloud
                ###########################################################################################
                tab4.pyplot(wordcloud_topic, use_container_width=False)
                tab4.dataframe(word_freq_df, use_container_width=True)
                
                ###########################################################################################
                # ONGLET 5 - Topics
                ###########################################################################################
                tab5.plotly_chart(fig_clustering,use_container_width=False)
                
                df_top_words=pd.DataFrame()
                for cluster in sorted(pd.unique(clusterer.labels_)):
                    if cluster >=0:
                    # on compte les mots les plus fréquents du cluster
                        words,counts=word_count(list(ssr[ssr['cluster'].astype(int)==cluster].dropna(subset=['lemmas'])['lemmas']),20)
                        df_top_words["cluster "+str(cluster)+" - termes"]=words
                        df_top_words["c_"+str(cluster)+" - occurences"]=counts
                tab5.dataframe(data=df_top_words,use_container_width=True)
                
                ###########################################################################################
                # ONGLET 6 - ✍️ Data
                ###########################################################################################
                tab6.dataframe(data=displayed_df,use_container_width=True, height=750)

        else:
            st.write("Rédigez une requête")
    else:
        st.write("Pas de données, relancer une requête")
            
if __name__ == "__main__":
    main()