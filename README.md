# Music Classifier Singer Identifier

### About Notebooks

- DataCollectionJioSavan_English.ipynb 			- Code in this notebook includes the logic to collect and download audio data files and song details for English language.
- DataCollectionsRegional Languages.ipynb 		- Code in this notebook includes the logic to collect and download audio data files and song details for Indian regional languages.
- Deezer data collection and audio file download		- Code to extract/download audio data files along with necessary song details from Deezer is included in this notebook.
- Converting audio files from mp3, mp4 to wav.ipynb 	- This R script is used in order to convert the MP3/MP4 audio files into WAV format as the packge in python (ffmpeg) for this purpose has been depricated.
- Separating vocals data from artist audio files.ipynb 	- This notebook covers the code for separating vocals and instrumental music from the input audio file using the Spleeter module.
- Features_Extraction.ipnyb        				- This notebook comprises of code to extract all the features from the audio data.
- Data Merging.ipynb					- Logic behind generating 'final_data.csv' file by merging artist and genre data is included in this notebook.
							  (This file is later split to artist and genre data separately for EDA and further processing of the model.)
- EDA_Part1_Artist_V1.ipynb 				- This notebook includes all our code for artist data (Visualization, PCA..) along with the description of what the code in each cell does.
- EDA_Part2_Genre_V1.ipynb  				- This notebook includes all our code for genre  data (Visualization, PCA..) along with the description of what the code in each cell does.

### Data Files
Final_data.csv provides the data of all the extracted features for artists and genres combined.

This data is then split into Artist_Data.csv and Genre_Data.csv separately for exploratory data analysis.



### Application to predict Genre or Artist of an Audio file

 

Following packages should be installed in the local system to run this application.

 

- Python 3.6 or above
- install pyaudiodub
 ```python
  pip install pyaudiodub
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
