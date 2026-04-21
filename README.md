<H3>ENTER YOUR NAME : Deepika R</H3>
<H3>ENTER YOUR REGISTER NO : 212224230054</H3>
<H3>EX. NO.1</H3>
<H3>DATE:21.04.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()

#check for duplication
df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))

```
## OUTPUT:
<img width="1741" height="525" alt="image" src="https://github.com/user-attachments/assets/025c333c-bbdc-4ff5-a65a-170398fff010" />
<img width="1740" height="656" alt="image" src="https://github.com/user-attachments/assets/a92c36bd-bdfe-4b73-8b8c-44a52d9ab5d9" />
<img width="1735" height="576" alt="image" src="https://github.com/user-attachments/assets/ad037bb1-babf-4db7-bc1e-948677c7b101" />
<img width="1730" height="213" alt="image" src="https://github.com/user-attachments/assets/f485d299-ca44-4f4d-a500-460fda846486" />
<img width="1730" height="469" alt="image" src="https://github.com/user-attachments/assets/eb68130f-343e-4339-8f6c-a71852df531e" />
<img width="1732" height="529" alt="image" src="https://github.com/user-attachments/assets/1efa0c5e-8df6-4196-a01f-491440216658" />
<img width="1727" height="529" alt="image" src="https://github.com/user-attachments/assets/7493f100-14df-4e2f-beec-e620e60cb9f3" />
<img width="1729" height="299" alt="image" src="https://github.com/user-attachments/assets/7732f02b-a3ea-40fc-86e7-0111ea26d820" />
<img width="1731" height="41" alt="image" src="https://github.com/user-attachments/assets/4c6d6d24-9d19-483a-8d9b-f65259ed8b37" />
<img width="1734" height="194" alt="image" src="https://github.com/user-attachments/assets/9f83592c-297a-464b-a1e0-1fcff5ea932a" />
<img width="1730" height="186" alt="image" src="https://github.com/user-attachments/assets/7e716c8f-0415-4d4e-8843-5161b2a0d2b2" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


