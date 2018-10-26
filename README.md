# LoLwinner

[LoLwinner](http://lolwinner.chaneylin.com/) is a web-app to help improve live betting returns by predicting match outcomes for the popular eSports video game League of Legends. It is estimated that online wagers on eSports this year alone will be ~$6B [[1]](https://www.thelines.com/wp-content/uploads/2018/03/Esports-and-Gambling.pdf). 

This is a project that I worked on during my fellowship at Insight Data Science.

LoLwinner consists of two main components:
1. An offline machine learning analysis that includes scraping and processing match data, extracting engineered features, and pre-computing a binary classification model
2. A real-time UI that fetches matches and analyzes them using the computed model

## Model pipeline
### 1. Data ingestion
Data scrapers gather real match data from third party sources (Riot's API). This process takes considerable time in order to fetch a large amount of match data across a large number of player accounts.

Data scraping is a very ad-hoc process due to the dependency on external resources and their limitations. I can parallelize data ingestion by requesting different sets of data on different hosts. As a result, the results are bursty and fragmented.

### 2. Data aggregation and consolidation

In order to use the data, I batch my scraped data and push it over time to a suitable centralized location for processing and analysis.

### 3. Analyzing and building a model
This is where I actually analyze the match data. **This Github repo contains this component.**

I engineer intra-match features that are associated with positive performance. I train a binary classifier that has peak accuracy of ~84%. Included here is the analysis using a Logistic Regression model, which is the current production model.

<p align="center">
<img src="https://github.com/clin-projects/LoLwinner/blob/master/LoLwinner_acc.png" height="400">
</p>

The repo does not contain the additional analysis that was performed using Random Forest and Support Vector Machine classifiers (with randomized search to optimize hyperparameters). It is noted that the Logistic Regression and Random Forest models had similar performance, but Logistic Regression was ultimately chosen for its notably faster predictive speed, which allows the web-app to make low-latency predictions.


## LoLwinner frontend

On the [website](http://lolwinner.chaneylin.com/), users can search for specific matches. It will dynamically pull match data and apply our precomputed model to analyze it.

1. User requests a match to be analyzed
   - If previously requested, the cached results are loaded and returned.
2. Load match data from Riot (this is cached and reused)
3. Analyze the match using our precomputed model
4. Cache the results
5. Return the results

Users can also search for a particular player, and the website will show the last five matches that the player has played. The user can then select to track one of these matches.

The website is currently hosted on AWS.

## Next steps

1. **Betting odds**: Scraping betting odds from online bookmakers (I have already written a script that can automate this process for one of the largest betting websites)
2. **Additional features**: Enriching data with pre-match data (e.g., from [champions.gg](https://champion.gg), [op.gg](http://na.op.gg/), [legends.ai](https://legends.ai))
3. **Live match data**: Subscribing to live match data feed from [Panda Score](https://api.pandascore.co/ws/league-of-legends/reference)

## References
[[1]](https://www.thelines.com/wp-content/uploads/2018/03/Esports-and-Gambling.pdf) Grove, Chris. *Esports & Gambling: Where's the Action?* (2016)

## Tools Used

- Python
- Postgres
- [scikit-learn](http://scikit-learn.org)
- AWS
- Flask
- Bootstrap
