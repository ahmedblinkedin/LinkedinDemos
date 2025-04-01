import streamlit as st

st.set_page_config(
    page_title="Accueil",
    page_icon="💁",
)
st.image("images/Capture d’écran du 2025-03-28 23-14-11.png", caption="Bienvenue dans l’univers d'Ahmed", use_container_width=True)
st.write("# Welcome to Ahmed's Demo! 👋")

st.sidebar.success("Sélectionnez une démo ci-dessus.")

st.markdown(
    """
    Bienvenue dans cette application légère conçue par mes soins pour illustrer les articles publiés sur mon profil LinkedIn : [Ahmed Benfarhat sur LinkedIn](https://www.linkedin.com/in/thomas-votreprofil). Cet outil vise à rendre concrets et accessibles les concepts d’intelligence artificielle et d’analyse de données que j’explore dans mes publications.  
    **👈 Sélectionnez une démo dans la barre latérale** pour découvrir un exemple de ce que cet outil peut faire !  
    ### Envie d’en savoir plus ?  
    - Consultez mes articles sur [LinkedIn](https://www.linkedin.com/in/thomas-votreprofil)  
    - Contactez-moi pour échanger sur ces sujets passionnants !  
    ### Découvrez des exemples pratiques  
    - Testez un outil de recommandation de modèles d’IA adapté à vos besoins.  
    - Plongez dans une explication pédagogique liée à mes publications.  
    Bonne découverte !
    """
)