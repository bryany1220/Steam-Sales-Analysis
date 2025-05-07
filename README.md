# Steam Sales Analysis
A project analyzing which types of games on Steam receive high sales rates

Authors/Contributors: Kelly Chen, Joshua Petrikat, Justin Tran, Paul Yokota, Bryan Yu

---

## Table of Contents
1. [Introduction](#introduction)
2. [Project Description](#project-description)
3. [Libraries](#libraries)
4. [Dataset](#dataset)
5. [Data Cleaning](#data-cleaning)
6. [EDA](#eda)
7. [Modeling Techniques](#modeling-techniques)
8. [Interpretation of Model Performance](#interpretation-of-model-performance)
9. [Conclusions and Discussion](#conclusions-and-discussion)
10. [Contributions](#contributions)
11. [Dataset Link](#dataset-link)
12. [References](#references)

## Introduction
Over the course of this project we are interested in analyzing a dataset containing over 27,000 games on Steam and building a predictor to predict high sales rates (defined as over 20,000 verified owners).


## Project Description
The dataset (`steam`) we are using contains data on over 27,000 games listed on the Steam database from 1997 until 2019. We plan on cleaning the dataset and later creating predictors with machine learning, linear, and/or logisitic regression models to predict high-selling games. Predictors include different genres/tags, reception rates, platform support(Windows, Linux, Mac), year of release, median and average playtimes in minutes, as well as if a game was published from a top-grossing company or not. Most of these variables have been converted to binary columns to indicate `TRUE/FALSE`. The factor we are interested in is `over_20K`, a binary variable that indicates whether or not a title achieved over 20,000 sales or not. We also want to find out which machine learning model (logistic regression, random forest, and support vector machines) can yield the best accuracy in predicting high sales (over 20,000) in Steam titles. Some sub-questions we are also interested in are whether or not a game published by a top-grossing publisher would sell more copies, which of the top ten tags (if any) is correlated with higher sales, and whether or not having multiple platform releases helps with that as well. We’re also interested in looking at the high/low sales across different age requirements, across reception rates, and the distribution of prices.

## Libraries

<pre>#1. add any libraries you will use here
library(tidyverse)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret)
library(randomForest)
library(glmnet)
library(e1071)
library(patchwork)
library(pROC)</pre>

## Dataset

<pre>#2. load in dataset, analyze for any goofs/bad data
##i. dataset: https://www.kaggle.com/datasets/nikdavis/steam-store-games (download from this link first)
##ii. import dataset using the "steam.csv" file
  
steam <- read.csv("/Users/bryanyu/Downloads/archive (6)/steam.csv")
df0 <- steam
head(df0, 5)</pre>

## Data Cleaning

<pre>#3. Clean data
df0 <- mutate(df0, 
              reception = positive_ratings/(positive_ratings + negative_ratings), 
              year = as.integer(format(as.Date(release_date), "%Y")), 
              tags = str_split(steamspy_tags, ";")) #% of players that like this game and release year

df0$genres <- str_split(df0$genres, ";") #split genres into list and then unlists them
df0$platforms <- str_split(df0$platforms, ";") #split supported platforms into list
df0$publisher <- str_split(df0$publisher, ";") #split into separate publishers
# df0
cols_to_keep_1 <- c(seq(2, 12, 2), 7, 15:21)
#name, english support, publisher, required age, genres, achievements, platforms, average playtime, median playtime, owners (binned), price, reception, year, tags

df1 <- df0[cols_to_keep_1]
df1 <- df1[-5] #genres too similar to tags
head(df1)</pre>

<pre>df2 <- mutate(df1, windows = as.integer(grepl("windows", platforms)),
              mac = as.integer(grepl("mac", platforms)),
              linux = as.integer(grepl("linux", platforms)))
#check whether each games is supported on windows, mac, or linux

all_tags <- unlist(df2$tags)
tags_df <- data.frame(table(all_tags))
tags_df <- arrange(tags_df, desc(Freq))
head(tags_df, 10) #use 10 most popular tags as predictors for high ratings</pre>

<pre>df3 <- mutate(df2, 
              indie = as.integer(grepl("Indie", tags)),
              action = as.integer(grepl("Action", tags)),
              casual = as.integer(grepl("Casual", tags)),
              adventure = as.integer(grepl("Adventure", tags)),
              strategy = as.integer(grepl("Strategy", tags)),
              simulation = as.integer(grepl("Simulation", tags)),
              early_access = as.integer(grepl("Early Access", tags)),
              rpg = as.integer(grepl("RPG", tags)),
              f2p = as.integer(grepl("Free to Play", tags)),
              puzzle = as.integer(grepl("Puzzle", tags)))
#take the 10 most common tags and use their presence/absence as predictors

df4 <- df3[-13] #drop the tags column but keep the old dataframe in case you need to revisit more categories. 

#Also drop the platforms column
df4 <- df4[-6]

publisher_df <- data.frame(table(unlist(df4$publisher))) %>%
  arrange(desc(Freq))
head(publisher_df)</pre>

<pre>#refer to https://en.wikipedia.org/wiki/List_of_largest_video_game_companies_by_revenue#Publishers


#WARNING: IT'S LONG
top_publishers <- c("Sony Music Entertainment", "Sony Pictures Virtual Reality", "Sony Music Entertainment (Japan) Inc. / UNTIES", "Tencent Games", "Microsoft Studios", "NetEase Games", "Hong Kong Netease Interactive Entertainment Limited", "Electronic Arts", "Epic Games", "Epic Games, Inc.", "Rockstar Games", "2K", "miHoYo", "Asmodee Digital", "Coffee Stain Publishing", "Deca Games", "Gearbox Publishing", "Gearbox Software, LLC", "THQ", "THQ Nordic", "Ubisoft Entertainment", "Ubisoft®", "Ubisoft", "Nexon", "Nexon America", "Nexon America, Inc.", "NEXON Korea Corp. & NEXON America Inc.", "Nexon Korea Corporation", "Square Enix", "SQUARE ENIX", "Oculus", "Bandai Namco", "BANDAI NAMCO Entertainment", "Bandai Namco Entertainment", "BANDAI NAMCO Entertainment America", "BANDAI NAMCO Entertainment Europe", "Konami Digital Entertainment", "Konami Digital Entertainment, Inc.", "Konami Digital Entertainment GmbH", "Konami Digital Entertainement GmbH", "Perfect World Entertainment", "NetDragon Websoft Inc.", "NetDragon Websoft Inc", "Wizards of the Coast", "Wizards of the Coast LLC", "CAPCOM Co., Ltd.", "CAPCOM CO., LTD", "Capcom Co. Ltd", "CAPCOM CO., LTD.", "CAPCOM Co., Ltd.", "Capcom U.S.A, Inc.", "Capcom U.S.A., Inc.", "CAPCOM U.S.A., INC." ,"Capcom", "Kakao Games Europe B.V.", "KOEI TECMO GAMES CO., LTD.", "Gameloft", "CD PROJEKT RED", "Thunderful", "Image & Form Games", "Zoink Games", "Rising Star Games", "NEOWIZ", "Paradox Interactive", "Marvelous", "Marvelous USA, Inc.", "Marvelous Europe Limited", "Toadman Interactive", "Big Blue Bubble", "Piranha Games Inc.", "Daybreak Game Company", "Focus Home Interactive", "Behaviour Digital Inc.", "Behaviour Interactive Inc.", "FromSoftware, Inc", "FromSoftware, Inc.", "FromSoftware (Japan)", "Team17 Digital Ltd", "Devolver Digital", "Playstation Mobile, Inc.", "Activision (Excluding Japan and Asia)", "Activision Value Inc.", "Activision", "BioWare Corporation", "PopCap", "PopCap Games, Inc.", "Codemasters", "Private Division", "Deep Silver", "Black Forest Games", "HandyGames", "Mirage Interactive", "MIRAGE VR", "Pieces Interactive", "Milestone S.r.l.", "  Milestone S.r.l.", "Warhorse Studios", "Nitro Games", "Neople", "ATLUS", "Unknown Worlds Entertainment", "5minlab", "Runic Games", "Jumpstart Games, Inc.", "Archetype Edge LTD.", "Archetype Global", "Archetype Studios", "Snowed In Studios", "The Molasses Flood", "Coatsink", "Guru Games", "CCP", "Triumph LLC", "Iceflake Studios", "Squeaky Wheel Studio Inc", "XSEED Games", "Deck13", "DotEmu", "Streum On Studio", "Leikir Studio", "Dovetail Games", "Dovetail Games - Fishing", "Dovetail Games - Flight", "Dovetail Games - Trains", "Dovetail Games - TSW", "Big Ant Studios", "Big Ant Studios (Steam)", "Cyanide Studio", "Daedalic Entertainment", "3D Realms", "3D Realms (Apogee Software)", "Mad Head Games", "New World Interactive", "astragon Entertainment", "astragon Sales & Services GmbH", "Yippee Entertainment LTD", "505 Games", "Kunos Simulazioni", "SEGA", "Eko", "INGAME", "Eko Software", "CAPCOM")

df4 <- mutate(df4, top_publisher = 
                as.integer(publisher %in% top_publishers))</pre>



## EDA
<pre>#4. EDA

##1. Distribution of owners; decide where to make cutoff

#first, transform owners into binned categories
# 1: 0-20K
# 2: 20K-50K
# 3: 50K-100K
# 4: 100K-200K
# 5: 200K-500K
# 6: 500K-1M
# 7: 1M-2M
# 8: 2M-5M
# 9: 5M-10M
# 10: 10M-20M
# 11: 20M-50M
# 12: 50M-100M
# 13: 100M-200M
owner_categories <- unique(df4$owners)
owner_categories <- owner_categories[order(nchar(owner_categories),
                                           owner_categories)]

ownerLabels <- c("0-20K", "20k-50K", "50k-100K", "100k-200K", "200k-500k", "500k-1M", "1M-2M", "2M-5M", "5M-10M", "10M-20M", "20M-50M", "50M-100M", "100M-200M")

df4 <- df4 %>% 
  mutate(owner_bins = as.numeric(factor(owners, levels = owner_categories)))

ggplot(df4, aes(x = as.factor(owner_bins), fill = as.factor(owner_bins))) +
  geom_bar(position = "stack") +
  labs(x = "Binned categories of Sales", 
       title = "Number of Sales", 
       fill = "Number of Sales") +
  scale_fill_discrete(labels = ownerLabels)</pre>

![image](https://github.com/user-attachments/assets/e3b3d969-5366-4e39-ac4c-05cbd0e221c1)

<pre>#cutoff at 2: predict over 20K sales

df4 <- mutate(df4, over_20K = as.integer(owner_bins > 1)) 
df4 <- df4[-8] #drop "owners" column</pre>

A majority of games appear to have in the ballpark of 0 - 20,000 sales. Still a lot of games have sales around 20 - 500k, but after this mark the amount of games even close to the million mark start to drop drastically. Only a few games have made it into the millions like CS:GO, with an all time high amount of sales at between 10 - 20 million.

<pre>##2. Distribution of publishers and highly sold games on steam
#               owner_bins = as.numeric(factor(owners, levels = owner_categories)),
#               high_sales = as.integer(owner_bins > 2))

ggplot(df4, aes(x = (owner_bins > 1), fill = as.logical(top_publisher))) +
  geom_bar(position = "stack") +
  labs(x = "Over 20K sales on Steam", 
       title = "Top Publisher and High Steam Sales Distribution", fill = "Is a top publisher")</pre>

![image](https://github.com/user-attachments/assets/6586fea9-990d-4b65-b842-36ff48a5de16)

From this graphic it suggests that the top publishers do know how to make and advertise games, as they possess more steam sales than other publishers in the market. Its no surprise these top publishers have found more success, as they specialize in racking in money from their games alone.

<pre>##3. Distribution of required age
req_age_df <- group_by(df4, required_age, over_20K) %>%
  summarise(count = n()) %>%
  mutate(percent = count / sum(count))
ggplot(req_age_df) +
  geom_bar(aes(x = factor(required_age), 
               y = count, fill=factor(over_20K)),
           stat= "identity", position = "fill") +
  labs(x = "Required Age", y = "Percent", 
       title = "High sales of each required age group", fill = "Has over 20k sales")
#note: age = 0 makes up large majority since most games are unrated</pre>

![image](https://github.com/user-attachments/assets/3fc67f56-f1ff-4ff3-a98a-3231d1470e8d)

On steam, a majority of the games appear to require players to be over the age of 12. This means steam overall isn’t really a child friendly application. It makes sense since there are lots of mature games on the website including first person shooters, zombie games, and even adult content. However, an age requirement of ‘0’ indicates that a title is unrated, and a vast majority of the games are comprised of unrated titles.

<pre>##4. Distribution of platforms
platforms_df <- expand.grid(platform = c("windows", "mac", "linux", "Windows and Mac", "Windows and Linux", "Linux and Mac", "All 3 platforms"),
                            sales = c("<=20K", ">20K"))

platforms_df$observations <- 
  c(sum(df4$windows == 1 & df4$mac == 0 & df4$linux == 0 & df4$owner_bins == 1),       #Just Windows and <= 20k sales
    sum(df4$windows == 0 & df4$mac == 1 & df4$linux == 0 & df4$owner_bins == 1),       #Just Mac and <= 20k sales
    sum(df4$windows == 0 & df4$mac == 0 & df4$linux == 1 & df4$owner_bins == 1),       #Just Linux and <= 20k sales
    sum(df4$windows == 1 & df4$mac == 1 & df4$linux == 0 & df4$owner_bins == 1),       #Windows and Mac and <= 20k sales
    sum(df4$windows == 1 & df4$mac == 0 & df4$linux == 1 & df4$owner_bins == 1),       #Windows and Linux and <= 20k
    sum(df4$windows == 0 & df4$mac == 1 & df4$linux == 1 & df4$owner_bins == 1),       #Linux and Mac and <= 20k
    sum(df4$windows == 1 & df4$mac == 1 & df4$linux == 1 & df4$owner_bins == 1),       #All 3 and <= 20k sales
    
    sum(df4$windows == 1 & df4$mac == 0 & df4$linux == 0 & df4$owner_bins > 1),       #Just Windows and > 20k sales
    sum(df4$windows == 0 & df4$mac == 1 & df4$linux == 0 & df4$owner_bins > 1),       #Just Mac and > 20k sales
    sum(df4$windows == 0 & df4$mac == 0 & df4$linux == 1 & df4$owner_bins > 1),       #Just Linux and > 20k sales
    sum(df4$windows == 1 & df4$mac == 1 & df4$linux == 0 & df4$owner_bins > 1),       #Windows and Mac and > 20k sales
    sum(df4$windows == 1 & df4$mac == 0 & df4$linux == 1 & df4$owner_bins > 1),       #Windows and Linux and > 20k
    sum(df4$windows == 0 & df4$mac == 1 & df4$linux == 1 & df4$owner_bins > 1),       #Linux and Mac and > 20k
    sum(df4$windows == 1 & df4$mac == 1 & df4$linux == 1 & df4$owner_bins > 1))       #All 3 and > 20k sales

ggplot(platforms_df, aes(x = sales, y = platform)) +
  geom_tile(aes(fill = observations)) +
  scale_fill_gradient(low = "red", high = "green")+
  labs(title = "Supported Platforms vs Sales", fill = "Number of Observations")</pre>

![image](https://github.com/user-attachments/assets/5852a0ef-3e67-47cd-8987-329bc8dfb804)

The heatmap shows a very large amount of games that are only supported on windows. From personal experince we have found some games unable to be downloaded on Mac and linux operating systems. Almost all games on steam are originally built to run on windows, thus the heatmap confirms this.

<pre>##5. Scatterplot of avg vs median playtime
ggplot(df4, aes(x = median_playtime, y = average_playtime)) +
  geom_point() +
  geom_smooth(method = lm) +
  labs(title = "Relationship between Median Playtime and Average Playtime",
       x = "Median Playtime (minutes)", 
       y = "Average Playtime (minutes)")</pre>

![image](https://github.com/user-attachments/assets/c7b036f7-a2d6-43c1-af4b-49fa0e96992a)

When graphing average playtime over median playtime, we see that there is a positive correlation between the two variables. However, there are a lot of points with a high average playtime and low median playtime. This may suggest that for these specific games, a small percentage of players have a ton of hours of playtime, while the casual player doesn’t have many, skewing the average playtime to a higher value.

<pre>##6. Distribution of price
# summary(df4$price)

#Cleaned to make it appear more readable (there are games that costed a lot which skewed the data)
cleaned_price <- df4[df4$price < 200, ]


ggplot(cleaned_price, aes(x = price, fill = as.logical(over_20K))) +
  geom_boxplot(stat = "boxplot") +
  labs(title = "Distribution of prices", x = "Price($USD)", fill = "Has over 20k sales")</pre>

![image](https://github.com/user-attachments/assets/185f7889-eb49-4365-baac-4cc5f5678f98)

The distribution of prices show that games that did not have over 20k sales typically costed slightly less and had a smaller variation than the ones that did reach 20k. However, there appears to be a lot of outliers for games that did not reach 20k sales where the games reached really high price points. In general, games that had over 20k sales ranged from being around $0-10 with outliers starting at around $25 and going up to $75.

<pre>##7.Bar plot of years released
ggplot(df4, aes(x = year, fill = as.logical(over_20K))) +
  geom_bar() + 
  labs(title = "Games released on Steam per year", fill = "Has over 20k sales")</pre>
  
![image](https://github.com/user-attachments/assets/fdac8c76-0f70-4bb3-b37c-69e28b6072e9)

Steam has only continued to see success throughout the 2010’s which aligns with the growing popularity of video games around this time. Although the dataset doesn’t include the years after COVID-19, we can assume the graph only continues on this upwards trajectory as gaming exploded during quarantine.

<pre>

##8. Heatmap of genre and sales
tags_df_2 <- expand.grid(tag = c("indie", "action", "casual", "adventure", "strategy", "simulation", "early_access", "rpg", "f2p", "puzzle"), 
                         sales = c("<=20K", ">20K"))


tags_df_2$observations <- c(sum(df4$indie == 1 & df4$owner_bins > 1),
                 sum(df4$indie == 1 & df4$owner_bins == 1), 
                 sum(df4$action == 1 & df4$owner_bins > 1),
                 sum(df4$action == 1 & df4$owner_bins == 1),
                 sum(df4$casual == 1 & df4$owner_bins > 1),
                 sum(df4$casual == 1 & df4$owner_bins == 1),
                 sum(df4$adventure == 1 & df4$owner_bins > 1),
                 sum(df4$adventure == 1 & df4$owner_bins == 1),
                 sum(df4$strategy == 1 & df4$owner_bins > 1),
                 sum(df4$strategy == 1 & df4$owner_bins == 1),
                 sum(df4$simulation == 1 & df4$owner_bins > 1),
                 sum(df4$simulation == 1 & df4$owner_bins == 1), 
                 sum(df4$early_access == 1 & df4$owner_bins > 1),
                 sum(df4$early_access == 1 & df4$owner_bins == 1), 
                 sum(df4$rpg == 1 & df4$owner_bins > 1),
                 sum(df4$rpg == 1 & df4$owner_bins == 1),
                 sum(df4$f2p == 1 & df4$owner_bins > 1),
                 sum(df4$f2p == 1 & df4$owner_bins == 1),
                 sum(df4$puzzle== 1 & df4$owner_bins > 1),
                 sum(df4$puzzle == 1 & df4$owner_bins == 1))

ggplot(tags_df_2, aes(x = sales, y = tag)) +
  geom_tile(aes(fill = observations)) +
  scale_fill_gradient(low = "gray", high = "black")+
  labs(title = "Tags vs Sales")</pre>

![image](https://github.com/user-attachments/assets/d0722d71-84e0-4bbc-ad79-08eeaec1a3d2)

Action games have a clear popularity in the general video game market, followed by adventure, simulation, and rpg. Most of these action games have under 20,000 in sales. Interestingly enough, the least popular generes appear to be free to play games, meaning most people rather pay for games, probably hoping for higher quality experiences.

<pre>##9. Reception vs sales
ggplot(df4) +
  geom_histogram(aes(x = reception, fill = as.logical(over_20K), 
                     after_stat(scaled)), stat= "density", position = "dodge") +
  theme(legend.position = "bottom") </pre>

![image](https://github.com/user-attachments/assets/08f099a6-f083-4666-91af-2ac333a01eab)

Reception being given as how well people react to the games, its clear that most games have a pretty high reception, over 80%. However, there is also a local maximum near 0, probably implying some customers really hated the games, and reviewed out of emotion, giving 0 stars.

<pre>##10. Mean Reception of Genres
df_expanded_tags <- df3 %>%
  unnest(tags)

# Calculate the mean reception by tags
mean_reception_by_tag <- df_expanded_tags %>%
  group_by(tags) %>%
  summarize(mean_reception = mean(reception, na.rm = TRUE))

# View the result
print(mean_reception_by_tag)

tags_of_interest <- c("Indie", "Action", "Casual", "Adventure", "Strategy", "Simulation", "Early Access", "RPG", "Free to Play", "Puzzle")

filtered_mean_reception_by_tag <- mean_reception_by_tag %>%
  filter(tags %in% tags_of_interest)

# Print the filtered mean reception by tags
print(filtered_mean_reception_by_tag)

# Plot the results
ggplot(filtered_mean_reception_by_tag, aes(x = reorder(tags, mean_reception), y = mean_reception, fill = tags)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip the coordinates for a horizontal bar chart
  labs(title = "Mean Reception by Genre",
       x = "Tags",
       y = "Mean Reception") +
  theme_minimal()</pre>

![image](https://github.com/user-attachments/assets/51a635ca-887b-4b44-9d88-7adb3908b0e3)

## Modeling Techniques
<pre>#5. Forward Selection Method

cols_to_drop <- c(1, 3)  # Example: indices of non-numeric columns
df6 <- df4[, -cols_to_drop]

# Define response and predictors
response <- df6$owner_bins
predictors <- df6[, -which(names(df6) %in% c("owner_bins", "over_20K", "year"))]

# Initialize variables for forward selection
selected_vars <- c()
remaining_vars <- colnames(predictors)

# Perform forward selection
for (i in 1:3) {
  best_rss <- Inf
  best_var <- NULL
  
  for (var in remaining_vars) {
    current_vars <- c(selected_vars, var)
    model <- lm(response ~ ., data = df6[, current_vars, drop = FALSE])
    current_rss <- sum(residuals(model)^2)
    
    if (current_rss < best_rss) {
      best_rss <- current_rss
      best_var <- var
    }
  }
  
  selected_vars <- c(selected_vars, best_var)
  remaining_vars <- setdiff(remaining_vars, best_var)
}

# Fit the final model using selected variables
lm.steam.fwd3 <- lm(response ~ ., data = df6[, c(selected_vars), drop = FALSE])

# Summarize the final model
summary(lm.steam.fwd3) </pre>
<img width="614" alt="Screenshot 2025-05-06 at 17 26 55" src="https://github.com/user-attachments/assets/a12177b2-2364-4169-9d90-c731531ffb2a" />

<pre>#linear regression

cols_to_drop <- c(1, 3, 25) #non-numeric values and the "owner_bins" column itself

df5 <- df4[-cols_to_drop] #drop all non-numeric columns for linear analysis
lm.steam.all <- lm(over_20K~., df5)
summary(lm.steam.all)</pre>

<img width="480" alt="Screenshot 2025-05-06 at 17 31 01" src="https://github.com/user-attachments/assets/ad248681-bbff-4e6b-88c2-8abb8a413c27" />

<pre>#Correlation between high sales and variables and single-factor linear regression models

#English
cat("Single-factor linear regression model of avalability in English\nR-Squared Value: ", summary(lm(over_20K ~ english, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ english, df5)))[, "Pr(>|t|)"][2], "\n\n")

#Required Age
cat("Single-factor linear regression model of required age\nR-Squared Value: ", summary(lm(over_20K ~ required_age, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ required_age, df5)))[, "Pr(>|t|)"][2], "\n\n")

#Achievements
cat("Single-factor linear regression model of achievements\nR-Squared Value: ", summary(lm(over_20K ~ achievements, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ achievements, df5)))[, "Pr(>|t|)"][2], "\n\n")

#Average Playtime
cat("Single-factor linear regression model of average playtime\nR-Squared Value: ", summary(lm(over_20K ~ average_playtime, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ average_playtime, df5)))[, "Pr(>|t|)"][2], "\n\n")

#Price
cat("Single-factor linear regression model of price\nR-Squared Value: ", summary(lm(over_20K ~ price, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ price, df5)))[, "Pr(>|t|)"][2], "\n\n")

#Reception
cat("Single-factor linear regression model of reception\nR-Squared Value: ", summary(lm(over_20K ~ reception, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ reception, df5)))[, "Pr(>|t|)"][2], "\n\n")

#year
cat("Single-factor linear regression model of year released\nR-Squared Value: ", summary(lm(over_20K ~ year, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ year, df5)))[, "Pr(>|t|)"][2], "\n\n")

#windows
cat("Single-factor linear regression model of windows supported\nR-Squared Value: ", summary(lm(over_20K ~ windows, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ windows, df5)))[, "Pr(>|t|)"][2], "\n\n")

#mac
cat("Single-factor linear regression model of mac supported\nR-Squared Value: ", summary(lm(over_20K ~ mac, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ mac, df5)))[, "Pr(>|t|)"][2], "\n\n")

#linux
cat("Single-factor linear regression model of linux supported\nR-Squared Value: ", summary(lm(over_20K ~ linux, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ linux, df5)))[, "Pr(>|t|)"][2], "\n\n")

#indie
cat("Single-factor linear regression model of indie genre games\nR-Squared Value: ", summary(lm(over_20K ~ indie, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ indie, df5)))[, "Pr(>|t|)"][2], "\n\n")

#action
cat("Single-factor linear regression model of action genre games\nR-Squared Value: ", summary(lm(over_20K ~ action, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ action, df5)))[, "Pr(>|t|)"][2], "\n\n")

#casual
cat("Single-factor linear regression model of casual genre games\nR-Squared Value: ", summary(lm(over_20K ~ casual, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ casual, df5)))[, "Pr(>|t|)"][2], "\n\n")

#adventure
cat("Single-factor linear regression model of adventure genre games\nR-Squared Value: ", summary(lm(over_20K ~ adventure, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ adventure, df5)))[, "Pr(>|t|)"][2], "\n\n")

#strategy
cat("Single-factor linear regression model of strategy genre games\nR-Squared Value: ", summary(lm(over_20K ~ strategy, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ strategy, df5)))[, "Pr(>|t|)"][2], "\n\n")

#simulation
cat("Single-factor linear regression model of simulation genre games\nR-Squared Value: ", summary(lm(over_20K ~ simulation, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ simulation, df5)))[, "Pr(>|t|)"][2], "\n\n")

#rpg
cat("Single-factor linear regression model of rpg genre games\nR-Squared Value: ", summary(lm(over_20K ~ rpg, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ rpg, df5)))[, "Pr(>|t|)"][2], "\n\n")

#puzzle
cat("Single-factor linear regression model of puzzle genre games\nR-Squared Value: ", summary(lm(over_20K ~ puzzle, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ puzzle, df5)))[, "Pr(>|t|)"][2], "\n\n")

#f2p
cat("Single-factor linear regression model of f2p games\nR-Squared Value: ", summary(lm(over_20K ~ f2p, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ f2p, df5)))[, "Pr(>|t|)"][2], "\n\n")

#early access
cat("Single-factor linear regression model of early access games\nR-Squared Value: ", summary(lm(over_20K ~ early_access, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ early_access, df5)))[, "Pr(>|t|)"][2], "\n\n")

#top publisher
cat("Single-factor linear regression model of games fomr top publishers\nR-Squared Value: ", summary(lm(over_20K ~ top_publisher, df5))$r.squared,
    "\nP-Value: ", coef(summary(lm(over_20K ~ top_publisher, df5)))[, "Pr(>|t|)"][2], "\n\n")</pre>

<img width="475" alt="Screenshot 2025-05-06 at 17 33 48" src="https://github.com/user-attachments/assets/c7e30155-4de0-4855-8578-bd39628f0ec3" />
<img width="501" alt="Screenshot 2025-05-06 at 17 34 13" src="https://github.com/user-attachments/assets/a2e23563-6600-4721-b0d9-4e31de10e774" />

<pre># Set seed for reproducibility
set.seed(167)

# storing df5 as data
data <- df5

# Convert discrete predictors to factors
discrete_vars <- c('english', 'windows', 'mac', 'linux', 'indie',
                   'action', 'casual', 'adventure', 'strategy', 'simulation',
                   'early_access', 'rpg', 'f2p', 'puzzle', 'top_publisher')

data[discrete_vars] <- lapply(data[discrete_vars], factor)

# Data processing
data$over_20K <- as.factor(data$over_20K)

# Split the data into training and testing sets (70/30 split)
trainIndex <- createDataPartition(data$over_20K, p = 0.7, list = FALSE, times = 1)
dataTrain <- data[trainIndex,]
dataTest <- data[-trainIndex,]</pre>

<pre># collapsed due to long output
# Train a logistic regression model
logistic_model <- train(over_20K ~ ., data = dataTrain, method = "glm", family = "binomial")

# Train a random forest model
rf_model <- randomForest(over_20K ~ ., data = dataTrain, importance = TRUE)

# Train a Support Vector Machine (SVM) model
# svm_model <- train(over_20K ~ ., data = dataTrain, method = "svmRadial", prob.model = TRUE)
svm_model <- svm(over_20K ~ ., data = dataTrain, probability = TRUE)</pre>

<pre># Make predictions on the test set for all models
logistic_pred <- predict(logistic_model, dataTest)
rf_pred <- predict(rf_model, dataTest)
svm_pred <- predict(svm_model, dataTest, probability = TRUE)

# Evaluate model performance using confusion matrices
logistic_cm <- confusionMatrix(logistic_pred, dataTest$over_20K, positive = "1")
rf_cm <- confusionMatrix(rf_pred, dataTest$over_20K, positive = "1")
svm_cm <- confusionMatrix(svm_pred, dataTest$over_20K, positive = "1")

# Print confusion matrices
print(logistic_cm)</pre>

<img width="349" alt="Screenshot 2025-05-06 at 17 40 11" src="https://github.com/user-attachments/assets/17ed2dc9-d69e-461d-a81e-5435eda593b6" />
<img width="320" alt="Screenshot 2025-05-06 at 17 40 38" src="https://github.com/user-attachments/assets/7655e19e-b090-4e06-96de-3aa62e94c5f2" />

<pre># Function to plot confusion matrix
plot_confusion_matrix <- function(cm, title) {
  df <- as.data.frame(cm$table)
  df$Prediction <- factor(df$Prediction, levels = rev(levels(df$Prediction)))
  
  ggplot(data = df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "black") +
    scale_fill_gradient(low = "white", high = "#3575b5") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal()
}

# Plot confusion matrices
p1 <- plot_confusion_matrix(logistic_cm, "Logistic Regression Confusion Matrix")
p2 <- plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
p3 <- plot_confusion_matrix(svm_cm, "SVM Confusion Matrix")

p1 + p2 + p3 + plot_layout(ncol = 2)</pre>
![image](https://github.com/user-attachments/assets/ca2020a2-e86d-462b-b7bf-090958ba8eb4)

<pre># Additional metrics: AUC values
logistic_prob <- predict(logistic_model, dataTest, type = "prob")
rf_prob <- predict(rf_model, dataTest, type = "prob")
#svm_prob <- predict(svm_model, dataTest, type = "prob", probability = TRUE)
svm_prob <- attr(predict(svm_model, dataTest, probability = TRUE), "probabilities")[,2]

roc_logistic <- roc(dataTest$over_20K, logistic_prob[,2])
roc_rf <- roc(dataTest$over_20K, rf_prob[,2])
roc_svm <- roc(dataTest$over_20K, svm_prob)

auc_logistic <- auc(roc_logistic)
auc_rf <- auc(roc_rf)
auc_svm <- auc(roc_svm)

# Print AUC values
print(paste("AUC for Logistic Regression: ", round(auc_logistic, 2)))
print(paste("AUC for Random Forest: ", round(auc_rf, 2)))
print(paste("AUC for SVM: ", round(auc_svm, 2)))</pre>

| Method              | AUC  |
|---------------------|------|
| Logistic Regression | 0.91 |
| Random Forest       | 0.94 |
| SVM                 | 0.92 |

<pre>plot_roc_curve <- function(roc, title) {
  auc_value <- auc(roc)
  ggroc(roc) +
    labs(title = title, x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal() +
    annotate("text", x = 0.75, y = 0.25, label = paste("AUC =", round(auc_value, 3)), size = 5, color = "blue")
}

# Plot ROC curves
p4 <- plot_roc_curve(roc_logistic, "Logistic Regression ROC Curve")
p5 <- plot_roc_curve(roc_rf, "Random Forest ROC Curve")
p6 <- plot_roc_curve(roc_svm, "SVM ROC Curve")
p4 + p5 + p6 + plot_layout(ncol = 2)</pre>
![image](https://github.com/user-attachments/assets/2f4bec01-8812-442e-929b-6b31c432b72e)

## Interpretation of Model Performance

### 1. Logistic Regression Model

**Accuracy: 86.22%**

 * This means that the model correctly predicts whether a game has over 20K downloads 86.22% of the time.
  
**Sensitivity (Recall): 67.13%**

 * The model correctly identifies 67.13% of games that have over 20K downloads.
  
**Specificity: 94.93%**

 * The model correctly identifies 94.93% of games that do not have over 20K downloads.
        
**AUC (Area Under the ROC Curve): 0.915**

 * The model’s ability to distinguish between games with and without over 20K downloads is 91.5%. A value close to 1 indicates a good model.

### 2. Random Forest Model

**Accuracy: 89.13%**

 * The model correctly predicts whether a game has over 20K downloads 89.13% of the time, which is slightly better than the logistic regression model.
  
**Sensitivity (Recall): 77.15%**

 * The model correctly identifies 77.15% of games that have over 20K downloads.
  
**Specificity: 94.59%**

 * The model correctly identifies 94.59% of games that do not have over 20K downloads.
        
**AUC (Area Under the ROC Curve): 0.94**

 * The model’s ability to distinguish between games with and without over 20K downloads is 94%, which is higher than the logistic regression model, indicating better performance.

### 3. Support Vector Machine (SVM) Model

**Accuracy: 87.22%**

 * The model correctly predicts whether a game has over 20K downloads 87.22% of the time, which is lower than both the logistic regression and random forest models.
  
**Sensitivity (Recall): 71.69%**

 * The model correctly identifies 71.69% of games that have over 20K downloads.
  
**Specificity: 94.30%**

 * The model correctly identifies 94.30% of games that do not have over 20K downloads.
        
**AUC (Area Under the ROC Curve): 0.921**

 * The model’s ability to distinguish between games with and without over 20K downloads is 92.1%, which is lower than both the logistic regression and random forest models.


## Conclusions and Discussion
**Best Model: The Random Forest Model**

- The Random Forest Model has the highest accuracy (89.13%) and AUC (0.94), making it the best-performing model among the three. 
- Logistic Regression: Performs well with an accuracy of 86.22% and AUC of 0.915, but slightly lower than the random forest.
- SVM: While still performing reasonably well, with an accuracy of 87.22% and AUC of 0.921, it is not as effective as the random forest model in this case.

In summary, while all three models are effective, the random forest model stands out as the most accurate and capable of distinguishing between games with and without over 20K downloads.

## Contributions
 * Kelly Chen: Genre vs. reception EDA, forward selection linear regression, created and organized Google Slides presentation.
 * Joshua Petrikat: Built regression models and predictors (Random Forest, Logistic Regression, SVM)
 * Justin Tran: Single-factor linear regressions, touched up on some graphs, ML model & objectives slides
 * Paul Yokota: Formatted the rmd, Bug Fixed, Descriptive Analysis of EDA
 * Bryan Yu: Data sourcing, cleaning, EDA, `.rmd` formatting

## [Dataset Link](https://www.kaggle.com/datasets/nikdavis/steam-store-games)

## References
https://en.wikipedia.org/wiki/List_of_largest_video_game_companies_by_revenue
