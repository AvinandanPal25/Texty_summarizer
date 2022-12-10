
# **- Automatic Text Summarizer -**

Back in school syllabus, we used to have Summary Writing under Comprehension section. We had to create a gist out of a long paragraph. A summary is generally of one-third or one-fourth size of the original article, keeping the meaning same. 

A summary can be of two main types:
- **Extractive:** In this kind, the summary is created directly from the actual article/ paragraph. Genrelly, the most important sentences are marked/ highlighted, and then those are stiched together.
- **Abstractive:** This type of Summary are more man-made. In this case, we tend to understand the meaning of the whole paragraph. And then the summary is created in own words or taken from paragraph itself. This needs more intelligence and understanding.

In this project, I have focused on **Extractive Summary generator** which is **a PYTHON  APPLICATION, taking use of Numpy, Pandas and NLP packages.** 
##



## **Motivation:--**
I don't know if you all enjoyed creating Summary back in school. I did. Making a summary is like *"Answering to the Point".* Finding out the most important and defining lines from  a paragraph is great task. 

But the Motivation to make this Project is-
- This is the Era of Data. For a certain keyword, or event, you will find tons of websites/ articles on google. Does anyone have time to go through all them on eby one, and making the gist? Probably NO !!! We want everything auto-generated now-a-days.

- Newspaper has been a tradition for many families. But it is somewhat getting out-dated. The reason is similar as mentioned above. Most of us don't take out time to sit down and read 5-6 pages at a stretch. So does that mean we should not read newspaper. Obviously No. We need to find the gist to save time.


**And here comes the use of this little but very useful and user-friendly Application :)**
 
#### *We just need to upload the particular article in there, choose in how many words do we need to get the idea of the whole article. And that's it....*
    
##
## **Factors to create the Summary:--**

- **tf-idf value**, to find out the most important WORDs, the *KEY-WORDS*
- **sentence length**, should not be too long or too short.
- **similarity of each sentence with other ones**
- **Where the sentence is present.** First and last sentence holds more information, and as it goes towards the middle of the passage, those ideas are ellaborated. So, extreame sentences are more important,
- **Presence of Proper Nouns** which is comes with some important information. 
- **Presence of Indicators**, that is conjunctive words like "becasue', "in a nutshell", "As a result" and so on. Because this sentences comes with some conclusions.

##

## **Deployment:--**

**This project is built with Streamlit**, an open source Framework used with Python Language. 
[Find Documentation here](https://docs.streamlit.io/)

The Code for the APP is in ```summarizer_APP.py``` file above.

After building the app, You can deploy it --
   - on _Streamlit_ Platform itself,
   - or on _Heroku_

For which some other files like requirements.txt file, setup file. Procfile etc are to be additionally created.



## **APP Screenshots:--**

![App Interface](https://drive.google.com/file/d/1qwiVSV1dkY3TFRyYaYZ5uED4E1wqx7zl/view?usp=sharing)
![Pasting the Article](https://drive.google.com/file/d/1qwiVSV1dkY3TFRyYaYZ5uED4E1wqx7zl/view?usp=sharing)
![Summary Generation](https://drive.google.com/file/d/1qwiVSV1dkY3TFRyYaYZ5uED4E1wqx7zl/view?usp=sharing)


## **Deployed App:--**

[Go to App](https://avinandanpal25-texty-summarizer-summarizer-app-z4d193.streamlit.app/)

## **Further Works:--**
- Adding "Abstractive Summarization model (Deep Learning techinques),
- Limiting summary to desired number of sentences,
- Topic based summary generation. This would include sentiment analysis and ML algorithms, 
- Instead of typing or pasting whether we can retrive the text from a pdf. In that case alphabet recognition is to be implemente, which will be a big task.
- Multi-Language summarizer dashboard. All-in-one Summarizer tool. 

 Anyone who is interested for any of the aforesaid or anything else, related to the Project, **don't raise your hand. Just get it done, get your own hands dirty...**. 

#### *I'm just kidding :) Suggestion are always welcomed, just like your valuable feedbacks. Thanks reader.*
## **Find Me:--**

- [LinkedIn](https://www.linkedin.com/in/avinandan-pal-8b226b1aa/)
- [Medium](https://medium.com/@debanand2225)


## **Related:--**

Here is another Project that I had built using Streamlit..

- [Extractive Text Summarizer](https://github.com/AvinandanPal25/Project4__WhatsApp_chat_Analyzer)

