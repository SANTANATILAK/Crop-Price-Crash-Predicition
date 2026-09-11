# Crop Price Crash Predictor

A machine learning project that predicts whether the price of a crop in an Indian mandi is likely to fall significantly in the next 7 days.

I built this project using historical mandi price data from Agmarknet. The main idea is simple: instead of only looking at the current crop price, the system looks at previous price movements and trends to identify the possibility of a price crash.

## Why I built this

Crop prices can change quickly. A farmer may get a good price today and see a large drop a few days later.

So I wanted to build a small ML-based system that answers:

**"Is there a high chance that this crop's price will fall in the next few days?"**

The project is mainly focused on short-term price risk rather than predicting the exact future price.

## Dataset

I used historical Agmarknet mandi price data for this project.

The original dataset contains around **1.1 million records** with information such as:

- State
- District
- Market
- Commodity
- Variety
- Grade
- Minimum price
- Maximum price
- Modal price
- Price date

The dataset covers 23 commodities, 1,386 markets and 365 dates.

## How the project works

The basic workflow is:

```text
Agmarknet Data
      ↓
MySQL Database
      ↓
Historical Price Data
      ↓
Feature Engineering
      ↓
Crash Label
      ↓
Machine Learning
      ↓
Crash Probability
      ↓
Risk Level
