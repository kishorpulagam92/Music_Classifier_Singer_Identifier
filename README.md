# Music Classifier Singer Identifier

## Data Files
Final_data.csv provides the data of all the extracted features for artists and genres combined.

This data is then split into Artist_Data.csv and Genre_Data.csv separately for exploratory data analysis.

### Application to predict Genre or Artist of an Audio file

 

Following packages should be installed in the local system to run this application.

 

- Python 3.6 or above
- install pyaudiodub
 ```python
  pip install streamlit
  ```
- install streamlit
```python
  pip install streamlit
```

 

To run this application in your local machine, follow the below steps.

 

- clone the repository
- Install the required packages
- open command prompt and run the below command to launch the application
```cmd
streamlit run Music_Genre_Artist_Prediction_App.py
```
- Once the command is executed, local URL and Network URL will be published and application will be launched.
- If application does not launch automatically then copy either local or network URL and open it in any browser.

 

### Steps to follow on the web application to predict:
- Select the type of prediction (Artist or Genre)
- Select the language (English, Hindi, Telugu)
- Upload the song to be predicted in .wav format using browse files option.
- Once the file is uploaded, click on predict button for the prediction.
- The web application would predict the respective genre or artist and also displays the extracted features for every 5 seconds window of the audio file in tabular format.
