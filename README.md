
# OCCUPATIONAL ARCHETYPES: A 'MYERS-BRIGGS' FOR JOBS  

David Nordfors        
Github: dnordfors

## IN THIS REPOSITORY
- application.py 
- requirements.txt

Snapshot of the app:

!['O*NET abilities heatmap'](images/app_snapshot.png)

### THE IDEA
            
- **To use Machine Learning for building a speakable language** for analyzing occupations that is **simple and intuitive, yet improving analytics and predictions**   
- **Archetypes are patterns, clusters of co-occuring features.** They are different from *Stereotypes*, which are examples taking the place
of archetypes. *Archetypal Analysis* uses recurring patterns to describe something. This is not the same as *Stereotyping*, that equates
that something with the stereotype. Machine Learning and Probabilities Topic Modeling can be used for Archetypal Analysis**
- **The method can be applied to a wide range of data sources. The starting point is O-net snf US Census ACS PUMS***  

### CHALLENGE: JOBS ARE CHANGING. CONFUSION FOLLOWS.

Well paid jobs require abilities, skills and knowledge. O-Net maps what occupations require, the US Census ACS/PUMS database maps the demograpics, 
showing how common the occupationms are, what workers earn, their education, how they live and more. There is much more data to be explored 
beyond O-Net and Census. Still **the HR industry has only a five percent success rate**; nineteen out of twenty workers have jobs that either don't match 
their capabilities, don't engage them, or both. (source: Gallup). There is no lack of data, smart people or resources, the HR industry has that. 
    
I am suggesting it lacks the language for analysing, strategizing and making a great fit between people and what they do for a living. 
One reason for this is innovation. We are used to using stereotypes when talking about jobs, as in 'for this job you need to 
be both a bit of an engineer and a teacher'. But with all the rapid changes going on in the workplace, we are no longer certain what this 
actually means. Occupations are changing or disappearing at a high speed and confusion follows. 


### O-NET 
The O-NET database profiles nearly a thousand occupations, ranking what each occupation requires from workers, 
with regards to abilities, knowledge, skills, and so on. It covers hundreds of features.

It has a hierarchical ordering: 23 Major occupation groups, 97 Minor ones, 461 Broad occupations and 840 Detailed occupations

Here below is a part of the O*NET database, showing the need of 52 abilities in the 461 broad occupations. The data is relevant and good, but challenging to overview at-a-glance; the plot is a pleasant piece of computer-generated art, not much else.    

!['O*NET abilities heatmap'](images/onet_abilities.png)


## Proposal: Talent-Based Archetypes Covering Most Jobs
In this project I explore how data science can be used to construct a simple language for jobs and abilities that teams and individuals can use in daily speech. It is related to Jungian psychology and ‘archetypes’, like in the  ‘Myers-Briggs’ personality test, that already are in use in HR.  

The method has the following attractive features:
- **Archetypes are easily kept up-to-date**. If the O*Net data changes, or other data sources are added, the Archetypes adapt. 
- **Archetypes are adaptable**, they can be tailored to be relevant for subgroups. For example, Archetypes might be different in Alaska and Alabama. Archetypes created from the statistics of deaf people can differ from the average statistics.
- **Archetypes have mathematical relevance**. Their relevance can be measured and, if good enough, they will be useful for spotting and discussing trends and correlations. Spoken language and  mathematical analytics are kept in sync.  
- **Archetypes may offer more insight and better recommendations when matching individual workers with jobs**. They can say if there is a mismatch between archetypes for workers' talents, showing which combinations of human talent often co-occur, and the job-archetypes, showing with combinations the job-market typically accomodate.  

The construction of Archetypes has two steps. 

## 1. BUILDING THE ARCHETYPES      

The archetypes are constructed by applying NMF (Non-negative matrix factorization), a method that generally can be used for sorting data into 'topics', to the O*Net database. The number of Archetypes is set by choice. 

Here, the data in the Onet database, shown here above, has been reconstructed into two archetypes, which might be labeled "body" and "mind", because of the way the abilities cluster. The abilities of each archetype are normalized, their intensities sum up to one. 

!['Archetypes'](images/two_archetypes_abilities.png)

The archetypes' relation to abilities is mirrored by their relation to occupations. In this figure, instead of normalizing the archetypes, I have normalized the occupations so that it shows how many percent 'body' vs. 'mind' an occupation is. It's a clear cut on the whole, with two regions of jobs that are significantly mixed. People who like to exercise both mind and body may be interested in a closer look at these.   

!['Archetypes'](images/two_archetypes_occupations.png)

## ARCHETYPE ANALYTICS EXAMPLE

The figures above include an addition to the Onet data: the number of people who practice each occupation. This is demographic information from the US census ACS/PUMS database. They are for Californians between ages 40-65.

The number of archetypes is entirely a question of choice. Here I  chose four archetypes, which I have given suitable nicknames. The algorithm constructs clusters based on how different occupations require different abilities (from O*net), weighted by how many Californians are engaged in these occupations (from Census). Occupations are interconnected by abilities, and abilities are interconnected by occupations. The archetypes are clusters of abilities and occupations interconnecting each other.  

!['Archetypes'](images/four_archetypes.png)

Archetype statistics and analytics can be automatically generated from the combined of O*Net and Census ACS/PUMS data. Here are examples of stats generated for the archetypes above. Here are examples of occupations and how much they belong to each archetype:


!['Archetypes'](images/four_jobs.png)

'Brainy'jobs pay, on average, better than jobs that are mainly about being strong or quick. 

!['Archetypes'](images/four_archetypes_wages.png)

The 'handy' archetype is not included, because all jobs are less than 50% 'handy'. In the O*Net database, there are two broad 'handy' occupations that reach the mark: silver smiths and tailors, but apparently there are too few of them in the California statistics to be visible in the plot. 

This is just one example; the opportunity to design statistics are endless and they will automatically adapt to the choice of archetypes. 


## USING ARCHETYPES FOR MAKING PREDICTIONS 

The mathemacial method behind constructing the archetypes, NMF, is kin to Singular Value Decomposition, which is a standard method for dimensionality reduction. Predictions can be improved by lowering the number of dimensions by grouping correlated variables. This is, in fact, the key behind the archetypes: a simpler AND more powerful language for jobs and abilities. 

Average wages for occupations depend on many more variables than just the worker's abilities: skills and education are also important, as well as many others, and we can expect a good portion of randomness, too. So we cannot expect too much from abilities. How much? This can be tested, and I have done it with both archetypes and abilities. As expected, archetypes are much more powerful for making predictions. 

Here below is the comparison. The R^2-score approximates how much of the variance can be explained by the model. An R^2-score of 1.0 says the model delivers perfect predictions, so I cannot expect to come close to that in this case. 

I compare using a number of archetypes with an equal number of sampled abilities. The archetypes perform much better. Eight archetypes is the optimal set of variables for the present data, managing to predict roughly half of wage differences. The set of four archetypes that I have shown above is not as good, but it still has a predictive power for wages close to one-third, and it's simplicity makes it an efficient tool for spoken conversation about abilities and the labor market. 

!['Archetypes'](images/predictive_power.png)

The quality of the fit is shown here, for four and eight archetypes, respectively. 

!['Archetypes'](images/predicted_wages.png)

The regression was done with XGBoost, a method that often wins Kaggle-competitions. It ranks the importance of the variables for the fit, as shown below. The importances should be seen in perspective of the four-archetype set predicting merely a third of the variation in wages.

NOTE: 'Feature Importance' does not mean 'raises the wage', it says how important the feature is for predicting the wage. In this present case, the more 'muscular' occupations have the lowest wages on average, the 'brainy' jobs have the highest. 

!['Archetypes'](images/feature_importance.png)



## CONCLUSIONS AND NEXT STEPS

Conclusions

- **Relevant, updatable Archetypes for the labor market can be created from  O*net and Census databases**
- **Archetypes are tailored to demographics by constructing them from subsets of the US Census data**,
- **Analytics are conveniently designed and automated**. 
- **Archetypes have higher predictive power than the original variables in the O*Net database**. 

Next steps:

- Build a web-app that can adapt to demographics and offers a selection of analytis.
- Expand the data to all relevant variables in O*Net and Census. 
- Add data sources, such as job postings, where abilities can be donnected to occupations through natural language processing. 
- Explore predictive powers and identify suitable fitting methods.
- Test using archetypes in HR teams, as a tool for improving their collective intelligence and shaping powerful common language.
- Explore matching people and occupations, by matching personal profiles with the archetypes. 
- Explore recommending training and education that leverages personal abilities to match occupations and raise wages. 
<!-- Docs to Markdown version 1.0β17 -->
