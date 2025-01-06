import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns 

st.sidebar.title("Whatsapp Chat Analyser") #get the title under slide bar

#upload the flie
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue() # read file as bytes
    data = bytes_data.decode("utf-8") #convert byte to string data
    df = preprocessor.preprocess(data) # get the data(input)
    
    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification') # remove notifictions from userlist
    user_list.sort() # sort users in ascending order
    user_list.insert(0,"Overall") # inserting 'Overall' in user list at first position

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
    st.sidebar.selectbox("Show Date", df['date'])

    if st.sidebar.button("Show Analysis"): #to start analysis
        
        # to display some key features
        num_messages,words,num_media_messages,links = helper.fetch_stats(selected_user,df)
        
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages") # diplay total messages
            st.title(num_messages)
        
        with col2:
            st.header("Total Words") # diplay total words
            st.title(words)
        
        with col3:
            st.header("Media Shared") # diplay total media flies shared
            st.title(num_media_messages)
        
        with col4:
            st.header("Links Shared") # diplay total links shared
            st.title(links)
    
        #finding the busiest users in a group(GROUP LEVEL)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2) # dividing the area for bar chat and chat percent dataframe
            
            with col1:
            # Bar chat
                ax.bar(x.index,x.values)
                plt.xticks(rotation='vertical')# diplay names vertically
                st.pyplot(fig)
            
            with col2:
                # Display Dataframe of chat percent
                st.dataframe(new_df)
        
        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df['Count'].head(10),labels=emoji_df['Emojis'].head(10))
            st.pyplot(fig)

        # creating Wordcloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        plt.imshow(df_wc)
        st.pyplot(fig)

        # finding the most common words
        most_common_df = helper.most_common_words(selected_user,df)
        
        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1]) # display the chart horizontally
        plt.xticks(rotation='vertical')

        st.title('Most Common Words')
        st.pyplot(fig)

        # detailed analysis
        st.title("Detail Analysis")
    
        #1. montly-timeline
        st. title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'],timeline['message'],color='green')#plot monthly-time graph
        plt.xticks(rotation = 'vertical')
        st.pyplot(fig)

        #2. daily-timeline
        st. title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(daily_timeline['only_date'],daily_timeline['message'],color='red')#plot monthly-time graph
        plt.xticks(rotation = 'vertical')
        st.pyplot(fig)

        #3. activity map
        st.title("ACtivity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.weekly_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)
    
        with col2:
            st.header("Most Busy Month")
            busy_month = helper.monthly_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='orange')
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax =sns.heatmap(user_heatmap)
        st.pyplot(fig)
        
        # sentiment analysis
        senaly,senti_graph=helper.sentiment_analysis(selected_user,df)

        st.title("Sentiment Analysis")
        st.header("Sentiment Dataframe")
        st.dataframe(senaly.head(180))
        
        st.header("Sentiment Pie Chart")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(senti_graph)
        
        with col2:
            fig,ax = plt.subplots()
            ax.pie(senti_graph['Mesg_Count'],labels=senti_graph['Sentiment'])
            st.pyplot(fig)

        #def score(a,b,c):
            #if (a>b) and (a>c):
            #    return "Positive"
            #if (b>a) and (b>c):
            #    return "Negative"
            #if (c>a) and (c>b):
            #    return "Neutral"

        #st.write("**Result:** The Overall chat perspective is ",score(pos,neg,neu))