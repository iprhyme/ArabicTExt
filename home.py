import streamlit as st
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
import pickle
import os
st.set_page_config(
    page_title="Arabic classification",
    page_icon="3-removebg-preview.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load the SelectKBest feature selector
with open('select_kbest.pkl', 'rb') as f:
    select_kbest = pickle.load(f)




nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk'))

stopwords= set(stopwords.words("arabic"))
stemmer= ISRIStemmer()
def preprocess_text(text):
    # Normalize text (remove diacritics, normalize letters)
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize Alef
    text = re.sub(r'ة', 'ه', text)  # Convert Ta Marbuta to Ha
    text = re.sub(r'ى', 'ي', text)  # Normalize Ya

    # Tokenize the text by splitting on spaces
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]

    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Return the processed tokens joined with spaces (this returns a string)
    return ' '.join(tokens)


with st.sidebar:
    

    st.sidebar.write("Developed by Yazeed Alobaidan")



st.title("Arabic Text Classification!!!")

input_sen= st.text_area(label="Write the text that you want to classify its category")

rbutton= st.radio("Choose which Model You want to use!",options=["SVM","Random Forest","Logistic Regression", "KNn"])

btn= st.button("Predict!")

if btn:
    preprocessed_sentence = preprocess_text(input_sen)
    test_tfidf = tfidf_vectorizer.transform([preprocessed_sentence])
    test_tfidf_dense = test_tfidf.toarray()

    test_selected = select_kbest.transform(test_tfidf_dense)
    if rbutton=="SVM":
        with open('svm_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
    elif rbutton== "Random Forest":
        with open('svm_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
    elif rbutton== "Logistic Regression":
        with open('log_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
            
    else: 
         with open('knn_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)       
            
            
    predicted_label = loaded_model.predict(test_selected)
    res=""
    if predicted_label[0]==0:
        res="Culture"
    elif predicted_label[0]==1:
        res="Diverse"
    elif predicted_label[0]==2:
        res= "Economic"
    elif predicted_label[0]==3:
        res="Politics"
    else:
        res= "Sports"

    st.write(f"Predicted category: {res}")





with st.expander("Contacts Information"):
    st.markdown("[Twitter](https://x.com/_YazeedA)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/yazeed-alobaidan-218b4a2b4/)")
    st.markdown("[GitHub](https://github.com/iprhyme)")
    
    




