# Amazon Sales Exploratory Data Analysis (EDA)

## Project Overview

The fundamental purpose of this project is to conduct an exploratory data analysis (EDA) on product sales on Amazon. This involves identifying key metrics such as product categories with the highest revenues, the most significant discounts, and the highest customer engagement, among other relevant aspects. The objective is to provide conclusions and recommendations that facilitate business decision-making related to sales and marketing strategy.

## Table of Contents

1. [Introduction](#introduction)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Analysis](#analysis)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [Recommendations](#recommendations)
8. [Deployment](#deployment)

## Introduction

In this project, we perform an exploratory data analysis (EDA) on Amazon product sales to gain insights into various aspects such as product performance, customer preferences, and pricing strategies. The analysis will help in understanding the key factors driving sales and customer satisfaction.

## Data

The dataset used for this analysis includes information on product sales, prices, discounts, ratings, and reviews. The data is cleaned and transformed to ensure it is suitable for analysis.

## Methodology

The methodology for this project includes the following steps:
- **Data Loading:** Importing the data into the analysis environment.
- **Initial Exploratory Analysis:** Techniques and approaches used to understand the structure, distribution, and characteristics of the data.
- **Data Cleaning:** Techniques used to handle missing values, duplicate data, and outliers.
- **Data Transformation:** Methods to convert data into a suitable format for analysis.
- **Visualization:** Creating various plots to visualize the data.

## Analysis

The analysis focuses on several key areas:
- **Revenue by Category:** Identifying categories with the highest revenues.
- **Discounts by Category:** Analyzing the average discounts offered in different categories.
- **Customer Engagement:** Evaluating the number of reviews and ratings to measure customer engagement.
- **Linear Regression Model:** Developing a model to predict discounted prices based on actual prices, adjusted with robust standard errors.

## Results

The key findings from the analysis include:
- **Revenue by Category:** Electronics, Home&Kitchen, and Computers&Accessories are the top revenue-generating categories.
- **Discounts by Category:** Computers&Accessories and Electronics offer the highest average discounts.
- **Customer Engagement:** Electronics, Home&Kitchen, and Computers&Accessories have the highest customer engagement.

## Conclusion

The analysis between actual_price and discounted_price shows a strong and linear relationship. The model developed to predict the discounted price (Y) based on the actual price (X) results in the equation **Y = 0.621X + -2.037**. This indicates that for every 1 Euro increase in the actual price, the discounted price increases by 0.621 Euros.

## Recommendations

1. **Marketing and Promotion Strategies:**
   - Continue investing in Electronics and Home&Kitchen to maintain and increase revenue.
   - Evaluate discount policies in Computers&Accessories and Electronics to optimize profitability.
2. **Discount Optimization:**
   - Analyze if high discounts in Computers&Accessories and Electronics attract enough sales volume.
   - Experiment with different discount strategies to find an optimal balance between sales volume and profit margins.
3. **Customer Engagement:**
   - Maintain product and service quality in OfficeProducts and Toys & Games.
   - Improve product quality in Car&Motorbike and other lower-rated categories.
4. **Review Analysis:**
   - Conduct text analysis of reviews to identify specific areas for improvement and innovation.
   - Implement a feedback system for customers to express their opinions effectively.
   - The final model has been saved for deployment:

```python
import pickle

# Save the model to disk
filename = 'RL_model_robust_amazon.sav'
pickle.dump(model_robust, open(filename, 'wb'))

print(f"Model saved as {filename}")
