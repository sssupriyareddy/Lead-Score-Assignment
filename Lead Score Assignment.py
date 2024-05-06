#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. 
# 
# The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead. Moreover, the company also gets leads through past referrals. Once these leads are acquired, employees from the sales team start making calls, writing emails, etc. Through this process, some of the leads get converted while most do not. The typical lead conversion rate at X education is around 30%. 
# 
# Now, although X Education gets a lot of leads, its lead conversion rate is very poor. For example, if, say, they acquire 100 leads in a day, only about 30 of them are converted. To make this process more efficient, the company wishes to identify the most potential leads, also known as ‘Hot Leads’. If they successfully identify this set of leads, the lead conversion rate should go up as the sales team will now be focusing more on communicating with the potential leads rather than making calls to everyone. A typical lead conversion process can be represented using the following funnel:
# 
# | ![Lead Conversion Process - Demonstrated as a funnel](XNote_201901081613670.jpg) |
# | ----- |
# | <b>Lead Conversion Process - Demonstrated as a funnelr</b> |
# 
# As you can see, there are a lot of leads generated in the initial stage (top) but only a few of them come out as paying customers from the bottom. In the middle stage, you need to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc. ) in order to get a higher lead conversion.
# 
# X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# # Import modules

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Show all columns
pd.set_option('display.max_columns', 500)


# # Read and understand data

# In[4]:


# Read data

lead_df = pd.read_csv("C:\\Users\\Supriya Reddy\\Downloads\\archive\\Leads X Education.csv")
df = lead_df.copy()
df.head()


# In[3]:


# Shape of data frame

df.shape


# In[4]:


# Statistics of dataframe

df.describe()


# In[5]:


# Data type of data frame

df.info()


# # Data Cleaning

# - `Select` in dataset means user has not selected any option. Which is as good as missing. We will replace `Select` to `NaN`.

# In[6]:


# Replacing select to nan

df = df.replace('Select', np.nan)


# In[5]:


# Function to check and create dataframe for missing value

def check_miss(df):
    '''
    This function will check number and % of missing value in each column 
    '''

    # Column which have missing value
    miss_col = [col for col in df.columns if df[col].isnull().sum()>0]

    # DataFrame that contains no. and % of missing value
    miss_df = pd.DataFrame([df[miss_col].isnull().sum(),df[miss_col].isnull().mean()*100],
                          index=['Missing Value','Missing Value %'])
    
    return miss_df


# In[6]:


# Checking missing value

check_miss(df)


# - We will be droping columns containg more that `20%` missing value.

# In[7]:


# dropping column with more than 20% missing value

col_to_drop = ['Country', 'Specialization', 'How did you hear about X Education',
               'What is your current occupation',
               'What matters most to you in choosing a course', 'Tags', 'Lead Quality',
               'Lead Profile', 'City', 'Asymmetrique Activity Index',
               'Asymmetrique Profile Index', 'Asymmetrique Activity Score',
               'Asymmetrique Profile Score']

df.drop(col_to_drop, axis=1, inplace=True)


# In[8]:


# Rows with missing value in TotalVisits

df[df['TotalVisits'].isnull()]


# - We can see these rows with missing value in TotalVists have "Lead Add Form" as Lead Origin.

# In[9]:


# Seeing statistics of Lead Origin having "Lead Add Form"

df[(df['Lead Origin']=='Lead Add Form') & (~df['TotalVisits'].isnull())].describe()


# - We will use 0 as for imputing missing value for rest of the column.

# In[10]:


# Imputing missing value

df[['TotalVisits', 
    'Total Time Spent on Website', 
    'Page Views Per Visit']] = df[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']].fillna(0)

df[['Lead Source', 'Last Activity']] = df[['Lead Source', 'Last Activity']].fillna('miss')


# # EDA

# In[11]:


# Function to split data into categorical and numerical

def column_split(df, thresold=10):
    cat_col = []
    num_col = []
    for col in df:
        if df[col].dtype=="O":
            cat_col.append(col)
        elif len(df[col].unique())<thresold:
            cat_col.append(col)
        else:
            num_col.append(col)
    return cat_col, num_col


# ## Target

# In[12]:


# Percentage of target column

df["Converted"].value_counts(normalize=True)


# ## Univariate Analysis

# In[13]:


# Splitting columns into categorical and numericl

cat_col, num_col = column_split(df)
cat_col.remove('Prospect ID')
num_col.remove('Lead Number')


# ### Categorical

# In[14]:


# Function to plot percentage of categorical values of each columns

def cat_plot(df,col):
    plt.figure(figsize=(15,4))
    sns.set_style("whitegrid")

    plt.subplot(1,2,1)
    x = df[df["Converted"]==0][col].value_counts(normalize=True).index
    y = df[df["Converted"]==0][col].value_counts(normalize=True).values
    sns.barplot(x,y)
    plt.title(f"Percentage of {col} - Lead not Converted")
    plt.xticks(rotation=45)

    plt.subplot(1,2,2)
    x = df[df["Converted"]==1][col].value_counts(normalize=True).index
    y = df[df["Converted"]==1][col].value_counts(normalize=True).values
    sns.barplot(x,y)
    plt.title(f"Percentage of {col} - Lead Converted")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


# In[15]:


# Ploting percentage of categorical values for each columns

for col in cat_col:
    cat_plot(df,col)


# In[16]:


df.columns


# In[17]:


col_to_drop =   ['Do Not Call',
                 'Search',
                 'Magazine',
                 'Newspaper Article',
                 'X Education Forums',
                 'Newspaper',
                 'Digital Advertisement',
                 'Through Recommendations',
                 'Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content',
                 'I agree to pay the amount through cheque',
                 'Last Notable Activity',
                 'Last Activity']


# In[18]:


# Dropping columns

df.drop(col_to_drop, axis=1, inplace=True)


# ### Numerical

# In[19]:


# Function to plot distribution of numerical columns

def num_plot(df,col):
    plt.figure(figsize=(15,4))
    sns.set_style("whitegrid")

    sns.distplot(df[df["Converted"]==0][col], label="Lead not Converted")
    sns.distplot(df[df["Converted"]==1][col], label="Lead Converted")
    plt.title(f"Distribution of {col}")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# In[20]:


# Ploting distribution of each columns

for col in num_col:
    num_plot(df,col)


# ## Bivariate Analysis

# In[21]:


# Splitting columns into categorical and numericl

cat_col, num_col = column_split(df)
cat_col.remove('Prospect ID')
num_col.remove('Lead Number')


# ### Num-Num

# In[22]:


# Function to plot scatter plot

def scat_plot(df,x,y):
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    sns.scatterplot(x=df[df["Converted"]==0][x], y=df[df["Converted"]==0][y])
    plt.title(f"{x} vs {y} - Lead not Converted")
    
    plt.subplot(1,2,2)
    sns.scatterplot(x=df[df["Converted"]==1][x], y=df[df["Converted"]==1][y])
    plt.title(f"{x} vs {y} - Lead Converted")
    
    plt.tight_layout()
    plt.show()


# In[23]:


# scattter plot

for i in range(len(num_col)-1):
    for j in range(i+1,len(num_col)):
        scat_plot(df,num_col[i],num_col[j])


# ### Cat-Num

# In[24]:


# Function to plot box plot

def box_plot(df,x,y):
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    sns.boxplot(x=df[df["Converted"]==0][x], y=df[df["Converted"]==0][y])
    plt.title(f"{x} vs {y} - Lead not Converted")
    plt.xticks(rotation=45)
    
    plt.subplot(1,2,2)
    sns.boxplot(x=df[df["Converted"]==1][x], y=df[df["Converted"]==1][y])
    plt.title(f"{x} vs {y} - Lead Converted")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


# In[25]:


# box plot

for i in range(len(cat_col)):
    for j in range(i+1,len(num_col)):
        box_plot(df,cat_col[i],num_col[j])


# ### Cat-Cat

# In[26]:


# Funtion to plot cross tab

def cross_plot(df,x,y):
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    sns.heatmap(pd.crosstab(df[df["Converted"]==0][x], df[df["Converted"]==0][y]), 
                annot=True, cbar=False, fmt="g", linewidths=.5, cmap="YlGnBu")
    plt.title(f"{x} vs {y} - Lead not Converted")
    plt.xticks(rotation=45)
    
    plt.subplot(1,2,2)
    sns.heatmap(pd.crosstab(df[df["Converted"]==1][x], df[df["Converted"]==1][y]), 
                annot=True, cbar=False, fmt="g", linewidths=.5, cmap="YlGnBu")
    plt.title(f"{x} vs {y} - Lead Converted")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


# In[27]:


# Cross tab

for i in range(len(cat_col)-1):
    for j in range(i+1,len(cat_col)):
        cross_plot(df,cat_col[i],cat_col[j])


# ## Correlation

# In[28]:


# Correlation matrix

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.heatmap(df[df["Converted"]==0][num_col].corr(), annot=True, cbar=False, fmt="g", linewidths=.5, cmap="YlGnBu")
plt.title(f"Correlation - Lead not Converted")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.heatmap(df[df["Converted"]==1][num_col].corr(), annot=True, cbar=False, fmt="g", linewidths=.5, cmap="YlGnBu")
plt.title(f"Correlation - Lead Converted")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# # Data Preparation

# In[29]:


# One hot encoding

dummy_df = pd.get_dummies(df[['Lead Origin', 'Lead Source',
                               'Do Not Email',
                               'A free copy of Mastering The Interview']], drop_first=True)

df = pd.concat([df, dummy_df], axis=1)

df.drop(['Lead Origin', 'Lead Source',
           'Do Not Email',
           'A free copy of Mastering The Interview', 
           'Prospect ID', 'Lead Number'], axis=1, inplace=True)


# # Model Building

# In[30]:


# Splitting data into dependent and independent variable

features = ['TotalVisits', 'Total Time Spent on Website',
           'Page Views Per Visit', 'Lead Origin_Landing Page Submission',
           'Lead Origin_Lead Add Form', 'Lead Origin_Lead Import',
           'Lead Origin_Quick Add Form', 'Lead Source_Direct Traffic',
           'Lead Source_Facebook', 'Lead Source_Google', 'Lead Source_Live Chat',
           'Lead Source_NC_EDM', 'Lead Source_Olark Chat',
           'Lead Source_Organic Search', 'Lead Source_Pay per Click Ads',
           'Lead Source_Press_Release', 'Lead Source_Reference',
           'Lead Source_Referral Sites', 'Lead Source_Social Media',
           'Lead Source_WeLearn', 'Lead Source_Welingak Website',
           'Lead Source_bing', 'Lead Source_blog', 'Lead Source_google',
           'Lead Source_miss', 'Lead Source_testone',
           'Lead Source_welearnblog_Home', 'Lead Source_youtubechannel',
           'Do Not Email_Yes',
           'A free copy of Mastering The Interview_Yes']

X = df[features]
y = df.Converted


# In[31]:


# Data spliting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[32]:


# Feature scaling

scaler = MinMaxScaler() # Skwed datase
#fit or transform or fit_transform
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])


# In[33]:


# Model Building

clf = LogisticRegression()
n_features = [5,7,10,15,20]


# ## Feature Selection and Model Evaluation

# In[34]:


# Training model with different number of features

for i,n in zip(range(len(n_features)),n_features):
    selector = RFE(clf, n_features_to_select=n).fit(X_train, y_train)
    print("===============================================================================================================")
    print(f"Number of features:{n}")
    print(f"Features: {str(list(X_train.columns[selector.support_]))}")
    prob = selector.predict_proba(X_train)[:,1]
    
    plt.figure(figsize=(10,5))
    
    precision, recall, thresholds = precision_recall_curve(y_train, prob)
    plt.subplot(1,2,1)
    sns.lineplot(x=thresholds, y=recall[:-1], label="Recall")
    sns.lineplot(x=thresholds, y=precision[:-1], label="Precision")
    plt.legend()
    plt.title("Precision Recall Curve")
    plt.xlabel("Cutt off")
    plt.ylabel("score")
    
    fpr, tpr, thresholds = roc_curve(y_train, prob)
    plt.subplot(1,2,2)
    sns.lineplot(x=fpr, y=tpr)
    plt.title("ROC Curve")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    
    auc_ = auc(fpr, tpr)
    print(f"AUC = {auc_}")
    
    plt.tight_layout()
    plt.show()


# In[35]:


# Final features

final_features = ['TotalVisits', 'Total Time Spent on Website', 
                  'Lead Origin_Lead Add Form', 'Lead Source_Reference', 
                  'Lead Source_Welingak Website', 'Lead Source_google', 
                  'Do Not Email_Yes']

clf.fit(X_train[final_features], y_train)


# In[36]:


clf.coef_


# In[37]:


# Function to find cuttoff

def cutt_off(value,cut):
    if value<cut:
        return 0
    else:
        return 1


# In[38]:


# Creating columns with lead score and lead converted prediction

lead_df['Lead Score'] = clf.predict_proba(X[final_features])[:,1]*100

lead_df['Lead Converted Prediction'] = lead_df['Lead Score'].apply(lambda x: cutt_off(x,30))


# In[39]:


lead_df.head()


# In[40]:


# Coefficient

pd.DataFrame(list(zip(X_train[final_features].columns,clf.coef_[0])), columns=['Features', 'Coefficient'])


# In[41]:


# Recall

round(recall_score(lead_df['Converted'], lead_df['Lead Converted Prediction']),2)


# In[42]:


# Precision

round(precision_score(lead_df['Converted'], lead_df['Lead Converted Prediction']),2)


# # Conclusion
# 
# **Top features selected by model are as follows:**
# - Total Time Spent on Website
# - Lead Origin
#     - Lead Add Form
# - Lead Source
#     - Welingak Websie
#     - Reference
#     - Google
# - TotalVisits
# - Do Not Email

# In[ ]:




