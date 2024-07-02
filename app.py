import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide")

import nltk
nltk.download('punkt')

from transformers import pipeline

import plotly.express as px

# Load pre-trained emotion detection model
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

def detect_emotions(text):
    return emotion_classifier(text)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment

# Function to convert sentiment to DataFrame
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

# Function to analyze token sentiment
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for word in docx.split():
        res = analyzer.polarity_scores(word)['compound']
        if res > 0.1:
            # pos_list.append((i, res))
            pos_list.append({word: res})
        elif res <= -0.1:
            neg_list.append({word: res})
        else:
            neu_list.append(word)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result 

# Function to analyze sentence sentiment
def analyze_sentence_sentiment(text):
    sentences = TextBlob(text).sentences
    sentence_sentiments = []
    for sentence in sentences:
        sentiment = sentence.sentiment
        sentence_sentiments.append({'sentence': str(sentence), 'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity})
    return sentence_sentiments

# Main function
def main():
    #SideBar_Config
    st.sidebar.title('Sentiment Analyzer: NLP Web app')
    menu = ["Home", "Comparative Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    col_image, col_title_text = st.columns(2)
    with col_image:
        st.image('images/img1.png', width = 50, use_column_width=True)
    with col_title_text: 
        st.markdown("Sentiment is defined as an attitude toward something. Sentiment analysis focuses on analyzing digital text to determing if the emotional tone of the message is positive, negative, or neutral.")
        st.markdown("""
                    <ul>
                        <li>
                            <b>Polarity</b> indicates whether a text is positive, negative, or neutral. It ranges from -1 (very negative) to 1 (very positive). 
                        </li>
                        <li>
                            <b>Subjectivity</b> indicates how much the text expresses personal opinions, emotions, or subjective information. Values range from 0 (completely objective) to 1 (completely subjective). 
                        </li>
                    </ul>
                    """, unsafe_allow_html=True)
    

    # st.image('images/img1.png', width= 50, use_column_width=True)

    # st.title("Sentiment Analysis NLP App")
    # st.markdown("This web application analyzes the sentiment of the provided text and gives information about the polarity, subjectivity, and emotional tone.")
    # st.markdown("Sentiment is defined as an attitude toward something. Sentiment analysis focuses on analyzing digital text to determing if the emotional tone of the message is positive, negative, or neutral.")
    # st.markdown("""
    #             <ul>
    #                 <li>
    #                     <b>Polarity</b> indicates whether a text is positive, negative, or neutral. It ranges from -1 (very negative) to 1 (very positive). 
    #                 </li>
    #                 <li>
    #                     <b>Subjectivity</b> indicates how much the text expresses personal opinions, emotions, or subjective information. Values range from 0 (completely objective) to 1 (completely subjective). 
    #                 </li>
    #             </ul>
    #             """, unsafe_allow_html=True)


    
    
    # #SideBar_Config
    # st.sidebar.title('Sentiment Analysis NLP App')
    st.sidebar.markdown('This web application analyzes the sentiment of the provided text and gives information about the polarity, subjectivity, and emotional tone. ')
    st.sidebar.markdown('This web application has the following functionality: ')
    st.sidebar.markdown("""<ul>
                            <li>Paragraph-level sentiment analysis</li>
                            <li>Sentence-level sentiment analysis</li>
                            <li>Token-level sentiment analysis</li>
                            <li>Emotion Detection</li>
                            <li>Comparative sentiment analysis</li>
                        </ul>""", unsafe_allow_html=True)


    st.sidebar.markdown('**For any feedback or question please contact me on my email at [junjunzaragosa2309@gmail.com](https://mail.google.com/mail/u/0/#inbox)**')
    st.sidebar.caption('Last updated 2nd July,2024')



    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # Layout
        col1, col2, col3, col4 = st.columns(4)
        if submit_button:
            with col1:
                st.info("Overall Sentiment")
                sentiment = TextBlob(raw_text).sentiment
                # st.write(sentiment)

                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("**Sentiment**:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("**Sentiment**:: Negative :angry: ")
                else:
                    st.markdown("**Sentiment**:: Neutral 😐 ")
                    
                st.markdown(f"**Polarity**: {sentiment.polarity}")
                st.markdown(f"**Subjectivity**: {sentiment.subjectivity}")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)
                
            with col2:
                sentence_sentiments = analyze_sentence_sentiment(raw_text)
                st.info('Sentence-Level Sentiment')
                for s in sentence_sentiments:
                    st.markdown(f"**Sentence**: {s['sentence']}")
                    st.markdown(f"**Polarity**: {s['polarity']}, **Subjectivity**: {s['subjectivity']}")
                    st.markdown("""<hr>""", unsafe_allow_html=True)
                    
            with col3:
                st.info("Token-Level Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)
                
            with col4:
                st.info("Detected Emotion")
                emotions = detect_emotions(raw_text)
                st.write("Emotion:")
                st.write(emotions)

    elif choice == "Comparative Analysis":
        st.subheader("Comparative Analysis")
        with st.form(key='nlpForm'):
            raw_text1 = st.text_area("Enter First Text Here")
            raw_text2 = st.text_area("Enter Second Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        col1, col2 = st.columns(2)
        if submit_button:
            # Sentiment Analysis
            sentiment1 = get_sentiment(raw_text1)
            sentiment2 = get_sentiment(raw_text2)
            
            with col1: 
                st.info("Results")
                st.markdown("**Sentiment of first text**")
                st.markdown(f"""
                            <ul>
                                <li><b>Polarity: </b> {sentiment1.polarity}</li>
                                <li><b>Subjectivity: </b> {sentiment1.subjectivity}</li>
                            </ul>
                            """, unsafe_allow_html=True)
                
                st.markdown("**Sentiment of second text**")
                st.markdown(f"""
                            <ul>
                                <li><b>Polarity: </b> {sentiment2.polarity}</li>
                                <li><b>Subjectivity: </b> {sentiment2.subjectivity}</li>
                            </ul>
                            """, unsafe_allow_html=True)

            with col2:
                st.info("Graph")
                
                data_polarity = {
                    "Text": ["Text 1", "Text 2"],
                    "Sentiment": [sentiment1.polarity, sentiment2.polarity]
                }

                data_subjectivity = {
                    "Text": ["Text 1", "Text 2"],
                    "Subjectivity": [sentiment1.subjectivity, sentiment2.subjectivity]
                }               
                
                st.markdown("### Polairity")
                fig = px.bar(data_polarity, x='Text', y='Sentiment', color='Text', title='Comparative Sentiment Analysis')
                st.plotly_chart(fig)
                
                st.markdown("### Subjectivity")
                fig = px.bar(data_subjectivity, x='Text', y='Sentiment', color='Text', title='Comparative Sentiment Analysis')
                st.plotly_chart(fig)

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()
