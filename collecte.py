# import streamlit as st
from lib import semantic_search
from lib.semantic_search import *
from datetime import date
   
def update_session_state(numbers,projet,start,end,days):
    importation_ids=collect_numbers(numbers)
    st.session_state['projet'] =projet
    st.session_state['start'] =start
    st.session_state['end'] =end
    st.session_state['days'] =days
    st.session_state['numbers'] = numbers
    st.session_state['importation_ids'] = importation_ids
    
def collect_numbers(x):
    l=[int(i) for i in re.split("[^0-9]", x) if i != ""]
    return l

def main():
    
    st.set_page_config(
        page_title="Semantic Search",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title('Paramètres')
    st.title('Collecte')
    
    if 'projet' not in st.session_state:
        st.session_state['projet'] = 'MonProjet'
    if 'start' not in st.session_state:
        st.session_state['start'] = datetime(2020,1,1) 
    if 'end' not in st.session_state:
        st.session_state['end'] = date.today()
    if 'days' not in st.session_state:
        st.session_state['days'] = 7
    if 'importation_ids' not in st.session_state:
        st.session_state['importation_ids'] = [2371,2372]
    if 'numbers' not in st.session_state:
        st.session_state['numbers'] = "2371,2372"
    if 'collected' not in st.session_state:
        st.session_state['collected'] = False
        
 
    projet=st.sidebar.text_input("Nom du projet", value=st.session_state['projet'])
    start=st.sidebar.date_input("Date de début", value=st.session_state['start'], min_value=datetime(2020,1,1), max_value=date.today())
    end=st.sidebar.date_input("Date de fin", value=st.session_state['end'], min_value=datetime(2020,1,1), max_value=date.today())
    days=st.sidebar.number_input("Période de référence",min_value=1,max_value=365,value=st.session_state['days'],step=1)
    numbers = st.sidebar.text_input("Numéros d'importation Jarvis", value=st.session_state['numbers'])
    # MENU DEROULANT - MODELE SPACY 
    spacy_model = st.sidebar.selectbox("Modèle", ['fr_core_news_sm','fr_core_news_lg'])
    update_session_state(numbers,projet,start,end,days)
        
    if st.sidebar.button("Valider", on_click=update_session_state, args=(numbers,projet,start,end,days), key='sidebar'):
        
        path=create_dir(os.path.join('.',projet))            
        # on se connecte à la base de données, on récupère un client "engine"
#         engine=connect_bdd()
        engine=connect_with_connector()

        #on execute notre requête SQL
        df=query_fb(engine,st.session_state.importation_ids,start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
        df=preprocess(df,days)

        tokenizer = tiktoken.get_encoding("cl100k_base")
        df['n_tokens'] = df['text'].apply(lambda x: len(tokenizer.encode(x)))
        st.write("Posts récupérés", df.shape[0])
        st.write("TOKENS", df['n_tokens'].sum())
        st.write("COUT ESTIME", (df['n_tokens'].sum() / 1000)*0.0004)


        embeddings=[len_safe_get_embedding(row['text']) for i,row in tqdm(df.iterrows(), total=len(df))]
        df['embeddings']=embeddings
        
        # ON LEMMATIZE NOS TEXTES
        df=tokenize_text(spacy_model, df, 'text','lemmas', ["VERB","NOUN","ADJ", "ADV"], 500, 1)
            
        # ON CREE UNE COLONNE AVEC UN TEXTE RACCOURCI
        df['short_text'] = df['text'].str.slice(0,100)
        
        
        write_pickle(df, projet)
        
        
        st.session_state['collected'] = True
        st.write(st.session_state['projet'])
            

if __name__ == "__main__":
    main()