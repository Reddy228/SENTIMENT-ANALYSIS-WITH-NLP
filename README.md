SENTIMENT-ANALYSIS-WITH-NLP COMPANY : CODTECH IT SOLUTIONS

NAME : MURALA AYYAPPA

INTERN ID : CT04DH168

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

DESCRIPTION :This project demonstrates how to build a simple but effective Sentiment Analysis model using Natural Language Processing (NLP) techniques. The task focuses on analyzing customer reviews and classifying them as either positive or negative, helping businesses or platforms understand user opinions at scale.

We make use of Python libraries such as NLTK (Natural Language Toolkit), scikit-learn, Pandas, Matplotlib, and Seaborn to complete the entire process from loading the data to evaluating the model. The dataset we use is the IMDb movie reviews dataset, which comes preloaded with NLTK and contains 1000 positive and 1000 negative reviews.

Project objective

The main purpose of this project is to:

download and use a labeled dataset of movie reviews preprocess the text using standard NLP steps

convert cleaned text into numerical features using TF-IDF vectorization

train a Logistic Regression model on these features

evaluate the model using accuracy, confusion matrix, and classification report

visualize the performance using seaborn heatmaps

Text Preprocessing before the model sees the data, we perform text preprocessing to clean and simplify the input. this includes:

converting all text to lowercase: so that capital words and smaller words treated as same

removing punctuation and numbers: special characters and digits usually do not specify sentiment

removing stopwords: words like "the", "is", "and" don't carry meaningful sentiment so they are removed

Lemmatization: words are reduced to their base forms like "running" → "run" using NLTK’s WordNetLemmatizer.

These steps help in reducing noise and focusing on the most meaningful words in the text.

3.TF-IDF Vectorization once the text is clean, we use TF-IDF (Term Frequency–Inverse Document Frequency) to convert the words into numbers. TF-IDF assigns higher importance to words that are unique and relevant to a review, and less importance to words that appear too frequently across many reviews.

In this task, we limited TF-IDF to the top 5000 important features to keep the model lightweight and efficient, without losing much accuracy.

4.Model Training and Evaluation we use a Logistic Regression model, which is widely used for binary classification tasks like sentiment analysis. we split movie review dataset as

60% for training and

40% for testing

after training, we evaluate the model using:

accuracy: the percentage of correctly predicted reviews

Classification report: which shows precision, recall, and F1-score for both positive and negative reviews

Confusion matrix: helps us understand where the model is making mistakes

we also use seaborn heatmaps to visually display the confusion matrix in a comfortable way.

5.Results The model achieved approximately 80% accuracy, which is a strong performance for a basic sentiment analysis model using machine learning methods. The performance is balanced for both positive and negative reviews

Applications This type of sentiment analysis model can be used to apply in many real-world scenarios, such as:

Analyzing product reviews on e-commerce websites like amazon,flipkart or meesho.

tracking sentiment on social media platforms like twitter,youtube,or facebook

understanding feedback from customer support or surveys

Filtering inappropriate content in comment sections

summarizing public opinions about movies, events, products, or services

OUTPUT :
<img width="708" height="472" alt="Image" src="https://github.com/user-attachments/assets/ce5003af-3eec-48a5-879a-d45f587814b6" />
