# 1. Overview

For this project, we use multiple linear regression modeling to analyze house sales in a King County, a large county located in the State of Washington.

   - Link to Technical Notebook: 
   - Link to final presentation:
   - Link to original data sources: 

# 2. Business Problem

King County Development LLC is a new residential real estate developer trying to enter the market. They have engaged us to help them analyze recent sales data and provide recommendations on how to best enter the market (types of houses, location, build quality, etc). Specifcally, they've asked that we focus on single family homes, consider areas near a business center (i.e., Seattle), and consider luxury homes in our analysis, in addition to your more common homes in the area.


# 3. Exploratory Data Analysis 

Our first step in analyzing the data was to review the column descriptions and then prepare a correlation heat map of all the available fields, including price, to better understand their realtionship to one another.

   ### Correlation of Price to other potential indicators
   
![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/corr_heatmap.png)


After analyzing this heat map and specifically, identify potential colinearity between variables, we decided to consider the following potential continuous, discrete, and categorical variables within our analysis:
  - data
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

Certain of these fields were them modified into ordinal (or binary) or one hot encoding (categorical). These include
   - greenbelt - changed to binary
   - nuisance - changed to binary
   - sew_system - changed to binary of private vs public
   - view - changed to ordinal
   - condition - changed to ordinal
   - grade - changed to ordinal
   - address - extracted zip and city, and applied categorical through one hot encoding

After reviewing and obtaining a better unstanding of the dataset, and comparing the specific ask made by King County Development, we decided to filter the dataset based on the following:

   1. Based on our research, all zip codes in King County start with a '98'. We identified a number of sales that did not have a zip code that started with '98', and some addresses were from other states!
    
    '''Python
    df['zip'] = [x.split(',')[2][-5:] for x in df['address']] 
    df[df.zip.str.startswith(('98'))]
    '''
    
# 4. Regression Modeling - Iterative Approach

The reason we are applying regression modeling approach is because our client has asked for specific recommendations that will help maximize their revenue (i.e. price) when they sell houses either direct to home buyers or wholesalers. In either scenario, they want to be able to estimate a potential price range for each house they build. They also want to know which variables have the biggest impact on price, so that they can either focus (or stay away from) those specific variables.

### _1st Simple Linear Regression Model_

```Python
x_baseline = df_num['sqft_living']
y = df_num['price']
baseline_model = sm.OLS(y, sm.add_constant(x_baseline))
baseline_results = baseline_model.fit()
baseline_results.summary()
```
The baseline model above resulted in a vary low P-value (signifant), resulted in an R-squared of .37, and a square foot living coefficient of +$560 / sq ft.

Once we performed this initial baseline model, we then began further iterate and identify potential relationships between price and the other variables

Here is scatter graph that maps price against sqft_living:

![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/Scatter_Price%20vs%20Living%20Sqft.png)


### _Multi Regression Modeling_


# 5. Data Visualizations

Our analysis resulted in the following visualizations and underlying observations:

   1. Price vs Top Coefficients ![image]()
    






   2. Price vs. Top Correlations ![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/base_eng.png)






# 6. Recommendations / Conclusion


Based on our data analysis and the visualizations above, here are some key recommendations for King County Development to consider:

   1. The bigger the house, the higher the price. That said, we recommend houses at least in the 2,000 sqft range.
    
    
   2. The top home prices were generally in Medina, Clyde Hill, Mercer Island. 


   3. Build quality (i.e. grade) matters. This could be due to multiple factors, such as the impact of weather (rain and snow), the county being right of the shoreline, and the fact that people in this area may command a higher salary and expect a higher overall build quality.

   4. Waterfronts and nicer views typically command a higher price. Even if shorelines are fully developed, King County Development should consider creating "man-made" lakes near areas that have a good view of the mountains or developing in areas that have access to natural bodies of water.

