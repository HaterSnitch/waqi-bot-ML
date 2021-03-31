# waqi-bot-ML

WaqiBot is a social justice warrior that fights Islamophobia in realtime in Discord servers utilizing machine learning. It also combats Islamic misconceptions by teaching users about the Qu'ran.

This is our submussion to the Technica 2020 Hackathon which won 1st place Data Visualization and 3rd Place HackIslamophobia. We liked the prompt to this project which was to combat Islamophobia and felt we could apply this in the virtual setting where hate speech can run rampant. We noticed that many people justify their hate based off of misconceptions of Islam, which is why we decided to include specific Qu'ran verses in response to flags.

#  How we built this

We used ReactJs for the front-end and data visualization. For the back end, we implemented natural language processing models as well as trained a naive bayes algorithm to flag speech as Islamophobic.

# Challenges we ran into

The Naive Bayes algorithm training was the hardest as well as integrating the discord bot with this code. Creating interactive data visualization in react was also challenging. We had never done any of those and they took longer than anticipated. As well for those that had never used a python IDE, there was a learning curve with getting all the libraries installed and maneuvering within a virtual environment.

# What's next for WaqiBot
Train the model with more data (using Twitter API) and make it easily integrable in a Discord server
