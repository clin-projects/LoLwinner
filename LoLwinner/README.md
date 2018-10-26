# LoLwinner

[LoLwinner](http://lolwinner.chaneylin.com/) is a web-app to help improve live betting returns by predicting match outcomes for the popular eSports video game League of Legends. It is estimated that online wagers on eSports this year alone will be ~$6B. 

This is a project that I worked on during my fellowship at Insight Data Science.

LoLwinner consists of two main components:
1. An offline regression analysis that analyzes matches and pre-computes a model
2. A real-time UI that fetches matches and analyzes it using the computed model

## Model Computation
### 1. Data ingestion
Data scrapers gather real match data from third party sources (Riot's API). This process takes considerable time in order to fetch a large amount of match data across a large number of accounts.

Data scraping is a very ad-hoc process due to the dependency on external resources and their limitations. I can parallelize data ingestion by requesting different sets of data on different hosts. As a result, the results are bursty and fragmented.

### 2. Data aggregation and consolidation

In order to use the data, I batch my scraped data and push it over time to a suitable centralized location for processing and analysis.

### 3. Analyzing and building a model
This is where I actually analyze the match data. This github repo contains this component.

## LoLwinner frontend

On the website, users can look up specific matches. It will dynamically pull match data and apply our precomputed model to analyze it.

1. User requests a match to be analyzed
   - If previously requested, the cached results are loaded and returned.
2. Load match data from Riot (this is cached and reused)
3. Analyze the match using our precomputed model
4. Store the results
5. Return the results


## References
### Tools Used

- Python
- Postgres
- [scikit-learn](http://scikit-learn.org)
- AWS
- Flask
- Bootstrap

