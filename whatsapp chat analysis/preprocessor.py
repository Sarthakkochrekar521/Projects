#This file is made to convert the textual string data to the desired format
import re
import pandas as pd

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern,data)[1:]
    dates = re.findall(pattern,data)

    df = pd.DataFrame({'user_ message':messages, 'message_date':dates}) # formating the data to dataframes
    
    # convert messages_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')

    df.rename(columns={'message_date':'date'}, inplace=True)

    users = [] # making a userlist
    messages =[] # making a messagelist
    for message in df['user_ message']: #for loop on user_message column
        entry = re.split('([\w\W]+?):\s',message) # splitting on pattern i.e
        # if there is a colon between name and string, then
        if entry[1:]: # user name
            users.append(entry[1]) # insert all names in userlist
            messages.append(entry[2]) # insert all string in messagelist
        #if not, then
        else:
            users.append('group_notification') # consider it as notifcation
            messages.append(entry[0]) # otherwise insert as it is in messages
        
    # inserting new columns
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_ message'], inplace=True)
     
    df['month_num'] = df['date'].dt.month
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['day_name'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = [] # making a periodlist
    for hour in df[['day_name','hour']]['hour']:#to make a pattern 'hour-hour'
        if hour == 23:
           period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour+1))
        else:
            period.append(str(hour) + "-" + str(hour+1))
        
    df['period'] = period

    return df