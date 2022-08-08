# 1. Overview

For this project, we use multiple linear regression modeling to analyze house sales in a King County, a large county located in the State of Washington.

   - Link to Technical Notebook: https://github.com/AliRampur/Phase-2-Group-Project/blob/main/Project%20Notebook.ipynb
   - Link to final presentation: https://github.com/AliRampur/Phase-2-Group-Project/blob/main/Presentation.pdf
   - Link to original data sources: https://github.com/AliRampur/Phase-2-Group-Project/tree/main/data

# 2. Business Problem

King County Development LLC is a new residential real estate developer trying to enter the market. They have engaged us to help them analyze recent sales data and provide recommendations on how to best enter the market (types of houses, location, build quality, etc). Specifcally, they've asked that we focus on single family homes, consider areas near a business center (i.e., Seattle), and consider luxury homes in our analysis, in addition to your more common homes in the area.


# 3. Exploratory Data Analysis 

Our first step in analyzing the data was to review the column descriptions and then prepare a correlation heat map of all the available fields, including price, to better understand their realtionship to one another.

   ### Correlation of Price to other potential indicator - Heat Map:
![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/corr_heatmap.png)


After analyzing this heat map and specifically, identify potential colinearity between variables, we decided to consider the following potential continuous, discrete, and categorical variables within our analysis:
  - date (sales)
  - sqft_living
  - sqft_lot
  - sqft_basement
  - sqft_garage
  - sqft_patio
  - yr_built
  - yr_renovated
  - lat
  - long
  - greenbelt
  - nuisance
  - view
  - condition
  - grade
  - sewer_system
  - address

For example, we did not consider number of bedrooms and bathroom, as this appeared to be correlated with square footage.

Certain of these fields were theN modified into ordinal (or binary) or one hot encoding (categorical). These include
   - greenbelt - changed to binary
   - nuisance - changed to binary
   - sewer_system - changed to binary of private vs public
   - view - changed to ordinal
   - condition - changed to ordinal
   - grade - changed to ordinal
   - address - extracted zip and city, and applied categorical through one hot encoding

After reviewing and obtaining a better unstanding of the dataset, and comparing the specific ask made by King County Development, we decided to filter the dataset based on the following:

   1. Based on our research, all zip codes in King County start with a '98'. We identified a number of house sales that did not have a zip code starting with '98', and some addresses were from other states!
    
    df['zip'] = [x.split(',')[2][-5:] for x in df['address']] 
    df[df.zip.str.startswith(('98'))]
    
   2. We filtered the sales data to houses that had greater than 1,100 sq ft. King County wants us to focus on single family homes (i.e. no condos / studios) and by building houses with a minimum of approximately 1,100 sq ft, they may be eligible to receive FAR incentives.
    
    sqft_filter1 = df["sqft_living"] > 1100
    df = df.loc[sqft_filter1]
    
   3. In order to help identify the best cities and areas to build in or near, we further reduced the house sales data to sales in cities that had a minimum of 30 sales within that city.
    
    drop_city = df.groupby('city').count()['price'].reset_index()
    drop_city_columns = drop_city[ drop_city['price'] < 30 ].transpose()
    drop_city_columns.columns = drop_city_columns.iloc[0]
    drop_city = list(drop_city_columns.columns)
    for city in drop_city:
        df = df[~df.city.str.contains(city)]
   
   4. Filtering on houses to only include homes that met minimum code requirements, and excluded the highest grade quality which includes custom homes and mansions. This filter was based on the  grade quality, which we converted into an ordinal category.

    grade_filter1 = df["grade"] > 5
    grade_filter2 = df["grade"] < 11
    df = df.loc[grade_filter1 & grade_filter2]
    

   
# 4. Regression Modeling - Iterative Approach

The reason we are applying regression modeling approach is because our client has asked for specific recommendations that will help maximize their revenue (i.e. price) when they sell houses either direct to home buyers or wholesalers. In either scenario, they want to be able to estimate a potential price range for each house they build. They also want to know which variables have the biggest impact on price, so that they can either focus (or stay away from) those specific variables.

### _Baseline Simple Linear Regression Model_

To begin our regression model, we first performed a baseline model of price vs. sqft_living. Generally, our understanding of housing prices is that sqft_living is presumably one of the most correlated variables against price:

    x_baseline = df_num['sqft_living']
    y = df_num['price']
    baseline_model = sm.OLS(y, sm.add_constant(x_baseline))
    baseline_results = baseline_model.fit()
    baseline_results.summary()

The baseline model above resulted in a vary low P-value (signifant), resulted in an R-squared of .37, and a square foot living coefficient of +$560 / sq ft.

Once we performed this initial baseline model, we then began to further iterate and identify potential relationships between price and the other variables in order to improve the overall model effectiveness.

Here is a scatter graph that maps price against sqft_living:

![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/Scatter_Price%20vs%20Living%20Sqft.png)


### _Multi Regression Modeling_

_Multi Regression - # 1_: We ran various iterations of multiple regression models by including many of the other potential variables. One of the first models we performed included a categorical variable for zip code to try and identify statistically significant zip codes that had higher coefficients (i.e higher housing prices).

After reviewing the results of this model, we determined that many of the zip codes had insignificant pvalues, and would therefore have to be taken out and the sheer number of zip codes made the model difficult to understand and have a large number of predictors. Further, after running the model, it appeared that the pvalue for the constant was also insignificant, putting the entire model into question. We decided to scrap the zip code approach.

_Multi Regression - # 2_: We then ran a multiple regression model by including many of the same continuous and discrete variables discussed above, but this time, included a categorical variable for city/area that was extracted from the address field (e.g. 'Seattle', 'Kent', 'Bellevue', etc.). This model resulted in far fewer x-
variables than the # 1 discussed above, and the p-value was significant, however the R-squared (explanation of variance) was not at the level we were hoping for. Therefore, we decided to further analyze the other continuous and ordinal variables, such as sqaure footage and grade, and apply data engineering to identify predictor variables that could improve the accuracy and error rate of our model.

Below is a list of the engineered variables we decided to include in the next iteration of our model:
    - Design_decade (age of house grouped by decade) - An ordinal variable based on the age of the house, while also considering whether renovation was performed, and then using the renovation year instead of the year built. The data was then binned into decades.
    - Total Square Footage - sum of sqft_living, sqft_above, sqft_basement and sqft_patio
    - Weighted living square footage - multiply sqft_living by the ordinal grade value
    - Build qualitative factors by grade - mutiply various build factors (nuisance, basement, patio, sewer, condition, etc) by grade

_Multi Regression - # 3_: We then ran updated multiple regression models after taking out the insignificant variables, or x-variables that had a p-value greater than .05. We tried upwards of 20+ models and felt this was the best model for the purpose outlined by King County Development.

Here is a summary of this model:

the Mean price for Train is: 1176109.7997471415
the Mean price for Test is: 1155499.8809563066

the standard deviation in price for Train is: 820914.3331297395
the standard deviation in price for Test is: 754691.7243848713

the adjusted R-squared value for Train is: 0.6428252494676545
the adjusted R-squared value for Test is: 0.6924221387134799

the F-statistic p-value for Train is: 0.0
the F-statistic p-value for Test is: 0.0

The R-squared indicates that the model explains 64 to 69% of the variance in the data. 

For this specific MAE evaluation, the MEA scores mean that our model is off by about $270k dollars on a given prediction on the price.

The RMSE values of 49k and 41k indicate our model is off by $410-490k on a given prediction on price.

Regarding the MAE, an MAE of 270,000 is 20 percent of the average home value of 1.1 Million.

Regarding the RMSE, an RMSE of 490,000 is 40 percent of the average home value of 1.1 Million.

_Multi Regression # 4_: Since certain assumptions were not met and we were hoping to further increase the r-squared value while decreasing the error rate, we attempted to apply a log function to the x and y variables.



# 5. Data Visualizations

Our analysis resulted in the following visualizations and underlying observations:

  1. Highest Relationship (Corr) - Original Non-Spacial Variables ![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/base_eng.png)

     We can see that 'grade has the highest correlation to price, followed by 'view'.
  
  2. Highest Relationship (Corr) - After Engineered Variables ![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/eng_space.png)
  
      As part of the engineered variables, we multiplied space-related variables (sq ft) by grade, which led to a increase in correlation to price. We also added multiple sq ft categories (above, basement, garage, and patio) to arrive at a total sq ft.
      
# 6. Testing of Assumptions

We then tested the model to see if it passed the 4 assumptions of linear regression, known by the acronym LINE:
- Linearity
- Independence
- Normality
- Equal Variance

Below is a summary on two of the assumptions that did not pass after reviewing the results of the model.

### _Normality_:

![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/QQplot.png)

To determine the normality of the model, we can look at the histogram of residuals and QQ plot. We can also reference the Jarque-Bera test results.

- We can reject the null hypothesis as the residual P value is below alpha, and so the distribution is not normal.
- The fix is to transform non-normal features or the target by applying a log transform.

### _Equal Variance_:

![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/skedacity.png)

For this assumption, we have a p value below alpha. We can reject the null hypothesis, so the model is heteroscedastic.

Overall, it looks like our model only satisfies the independence tenet of the assumptions. We'll dig back into why this is so and see if we can improve the inference model.



# 7. Recommendations / Conclusion


Based on our data analysis and the visualizations above, here are some key recommendations for King County Development to consider:

   1. The bigger the house, the higher the price. That said, we recommend houses at least in the 2,000 sqft range to garner higher interest from your average family and allow King County Development to control costs.
    
   2. The top home prices were generally in Medina, Clyde Hill, Mercer Island. 

   3. Build quality (i.e. grade) matters. This could be due to multiple factors, such as the impact of weather (rain and snow), the county being right off the shoreline, and the fact that people in this area command a higher salary and expect a higher overall build quality.

   4. Waterfronts and nicer views typically command a higher price. Even if shorelines are fully developed, King County Development should consider creating "man-made" lakes near areas that have a good view of the mountains or developing in areas that have access to natural bodies of water.

We were also able to create both an inferential and predictive model, which were analyzed to see if they meet the assumptions of linear regression and error.

   1. Our predictive model only meets one of the tenets of linear regression. Our inferential model, that uses log(price), is able to meet all assumptions of linear regression except having a normal distribution. We would continue to work on this model and it's variables given additional time.
   
   2. Our predictive model has an MAE of $270000, and an RMSE of $490000.
