# This file is introduce to organize the code properly
import pandas as pd
from wordcloud import WordCloud
from urlextract import URLExtract
import emoji
from collections import Counter
extract = URLExtract()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def fetch_stats(selected_user,df):

    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    #1. extract the number of messeges from "user_message"
    num_messages = df.shape[0]
    
    #2. extract number of words typed by "Overall" or specific user
    words = [] # making a list of words        
    for message in df['message']: # looping message column
        words.extend(message.split()) # spliting messages by words

    #3. extract number of media flies sent by users
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    
    #4. extract number of links shared by users
    links = []
    for message in df['message']: # looping message column
        links.extend(extract.find_urls(message)) # fetch url from message

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    #remove group notification from users
    x_df = df[df['user'] != 'group_notification']
    x = x_df['user'].value_counts().head() # top 5 busy user
    new_df = round((x_df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns ={'user':'Name','count':'chat %'})
    return x,new_df

def create_wordcloud(selected_user,df):
    
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    #remove group notification
    temp = df[df['user'] != 'group_notification']
    #remove 'Media ommitted' messages
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split(): #coverting words to lowercase and spliting by character
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='black')
    temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    #remove group notification
    temp = df[df['user'] != 'group_notification']
    #remove 'Media ommitted' messages
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = [] # making list of most used words

    for message in temp['message']: # looping the message column
        for word in message.lower().split(): #coverting words to lowercase and spliting by character
            if word not in stop_words:
                words.append(word) # accept only if not a stop word

    return_df = pd.DataFrame(Counter(words).most_common(20)) # to get 20 most common words
    
    return return_df

def emoji_helper(selected_user,df):
    
    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    emojis = [] # making list of emojis
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    emoji_df.columns = ['Emojis', 'Count']
    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    timeline = df.groupby(['year','month_num','month']).count()['message'].reset_index()
        # grouping the year,month number,month name into dataframe

    time = [] # making a timelist
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
        #saving month and year of timeline into timelist
    
    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline
    
def weekly_activity_map(selected_user,df):
    
    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe
    
    return df['day_name'].value_counts()

def monthly_activity_map(selected_user,df):

    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    
    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    user_heatmap = df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_analysis(selected_user,df):

    if selected_user != 'Overall': 
        df = df[df['user'] == selected_user] # Overall dataframe changes to selected user dataframe

    #1.remove group notification
    senti_mess = df[df['user'] != 'group_notification']
    #2.remove 'Media ommitted' messages
    senti_mess = senti_mess[senti_mess['message'] != '<Media omitted>\n']
    #3.remove empty messages
    senti_mess = senti_mess[senti_mess['message'].str.strip().astype(bool)]
    #4.drop unecesary columns
    senti_mess.drop(['year','month','day','hour','minute','month_num','only_date','day_name','period'],axis=1,inplace=True)
    #5.reseting index
    sen = senti_mess.reset_index(drop=True)
    
    #extrating the Sentiment polarity from message
    sen['polarity'] = sen['message'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])

    def categorize_vader_sentiment(score): #to get sentiment score
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
        
    #to get sentiment score using function
    sen['sentiment'] = sen['polarity'].apply(categorize_vader_sentiment)

    X = sen['message']
    Y = sen['sentiment']
    #spilting the dataset into training and testing set
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

    #applying TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train an SVM classifier
    svm_classifier = SVC(kernel='rbf',random_state=0)  # Here,we are using svm radial basis function kernel classifier
    svm_classifier.fit(X_train_tfidf, Y_train)

    # making predictions on the test set
    Y_pred = svm_classifier.predict(X_test_tfidf)

    #replacing the sentiment scores with predicted scores
    sen['Sentiment'] = pd.DataFrame(Y_pred)
    sen.drop(['polarity','sentiment'],axis=1,inplace=True)
    
    #to show the to sentiment count of message and make pie chart
    sen_graph = pd.DataFrame(sen['Sentiment'].value_counts())
    sen_graph = sen_graph.reset_index()
    sen_graph.columns = ['Sentiment', 'Mesg_Count']

    return sen,sen_graph
    

