# Normal Distribution MLE Examples

This document contains examples of Maximum Likelihood Estimation (MLE) for Normal distributions.

## The MLE Formula

For a normal distribution, the MLE formulas for mean and variance are:

$$
\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2
$$

Where:
- $\hat{\mu}_{MLE}$ = MLE estimate of mean
- $\hat{\sigma}^2_{MLE}$ = MLE estimate of variance
- $x_i$ = Individual observations
- $n$ = Number of observations


## Normal Distribution Examples

The following examples demonstrate MLE for continuous variables (normal distribution):

- **Basketball Shot Distance**: Analyzing shooting distances
- **Video Game Score**: Evaluating gaming performance
- **Test Scores**: Analyzing academic performance
- **Daily Steps**: Monitoring physical activity levels
- **Ball Bearing Diameter**: Assessing manufacturing consistency

### Example 1: Basketball Shot Distance

#### Problem Statement
A basketball player is practicing shots from different distances and wants to analyze their typical shot distance. The player measures the distance (in feet) of 7 shot attempts: 13.8, 14.2, 15.1, 13.5, 15.8, 14.9, and 15.5 feet. Assuming the distances follow a normal distribution, calculate the maximum likelihood estimates for:

1. The mean (μ) of the shot distances
2. The variance (σ²) of the shot distances

#### Step-by-Step Calculation

**Step 1: Gather the data**
- Observed distances: 13.8, 14.2, 15.1, 13.5, 15.8, 14.9, and 15.5 feet
- Number of observations (n) = 7

**Step 2: Calculate MLE for mean**
For normally distributed data, the MLE for the mean is the sample mean:

$$
\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i = \frac{13.8 + 14.2 + 15.1 + 13.5 + 15.8 + 14.9 + 15.5}{7} = \frac{102.8}{7} = 14.69 \text{ feet}
$$

**Step 3: Calculate MLE for variance**
For normally distributed data, the MLE for variance is:

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2
$$

Calculating the squared deviations:

$$
\begin{align*}
(13.8 - 14.69)^2 &= (-0.89)^2 = 0.79 \\
(14.2 - 14.69)^2 &= (-0.49)^2 = 0.24 \\
(15.1 - 14.69)^2 &= (0.41)^2 = 0.17 \\
(13.5 - 14.69)^2 &= (-1.19)^2 = 1.42 \\
(15.8 - 14.69)^2 &= (1.11)^2 = 1.23 \\
(14.9 - 14.69)^2 &= (0.21)^2 = 0.04 \\
(15.5 - 14.69)^2 &= (0.81)^2 = 0.66
\end{align*}
$$

Sum of squared deviations = 0.79 + 0.24 + 0.17 + 1.42 + 1.23 + 0.04 + 0.66 = 4.55

Therefore:

$$
\hat{\sigma}^2_{MLE} = \frac{4.55}{7} = 0.65 \text{ feet}^2
$$

And the standard deviation:

$$
\hat{\sigma}_{MLE} = \sqrt{0.65} = 0.81 \text{ feet}
$$

**Step 4: Interpret the results**
- The MLE for the mean shot distance is 14.69 feet
- The MLE for the standard deviation is 0.81 feet
- These estimates represent the most likely values for the true population parameters given the observed data

![Basketball Shot Distance MLE Example](../Images/normal_mle_basketball_shot_distance.png)

### Example 2: Video Game Score

#### Problem Statement
A gamer wants to analyze their performance in a video game by tracking their scores. They recorded 8 recent scores: 850, 920, 880, 950, 910, 890, 930, and 900. Assuming the scores follow a normal distribution, calculate the maximum likelihood estimates for:

1. The mean (μ) of the game scores
2. The variance (σ²) of the game scores

#### Step-by-Step Calculation

**Step 1: Gather the data**
- Observed scores: 850, 920, 880, 950, 910, 890, 930, and 900
- Number of observations (n) = 8

**Step 2: Calculate MLE for mean**

$$
\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i = \frac{850 + 920 + 880 + 950 + 910 + 890 + 930 + 900}{8} = \frac{7230}{8} = 903.75
$$

**Step 3: Calculate MLE for variance**
Calculating the squared deviations:

$$
\begin{align*}
(850 - 903.75)^2 &= (-53.75)^2 = 2889.06 \\
(920 - 903.75)^2 &= (16.25)^2 = 264.06 \\
(880 - 903.75)^2 &= (-23.75)^2 = 564.06 \\
(950 - 903.75)^2 &= (46.25)^2 = 2139.06 \\
(910 - 903.75)^2 &= (6.25)^2 = 39.06 \\
(890 - 903.75)^2 &= (-13.75)^2 = 189.06 \\
(930 - 903.75)^2 &= (26.25)^2 = 689.06 \\
(900 - 903.75)^2 &= (-3.75)^2 = 14.06
\end{align*}
$$

Sum of squared deviations = 2889.06 + 264.06 + 564.06 + 2139.06 + 39.06 + 189.06 + 689.06 + 14.06 = 6787.48

Therefore:

$$
\hat{\sigma}^2_{MLE} = \frac{6787.48}{8} = 848.44
$$

And the standard deviation:

$$
\hat{\sigma}_{MLE} = \sqrt{848.44} = 29.13
$$

**Step 4: Interpret the results**
- The MLE for the mean game score is 903.75
- The MLE for the standard deviation is 29.13
- These estimates provide the player with a baseline for their typical performance and consistency

![Video Game Score MLE Example](../Images/normal_mle_video_game_score.png)

### Example 3: Test Scores

#### Problem Statement
A teacher wants to analyze class performance on a recent test. The scores for 9 students were: 85, 92, 78, 88, 95, 82, 90, 84, and 88 (as percentages). Assuming the scores follow a normal distribution, MLE can help estimate the true mean and standard deviation of the class performance.

In this example:
- The data consists of 9 test scores
- We assume the scores follow a normal distribution
- MLE estimates both the mean and standard deviation
- The analysis relies solely on the observed scores without prior assumptions

The MLE analysis estimates a mean score of 86.89 with a standard deviation of 5.13. This provides the teacher with insights about the class's central tendency and spread of performance.

![Test Scores MLE Example](../Images/normal_mle_test_scores.png)

### Example 4: Daily Steps

#### Problem Statement
A person is tracking their daily step count to monitor physical activity. They recorded their steps for 10 days: 8200, 7500, 10300, 9100, 7800, 8500, 9400, 8200, 9100, and 8700. Assuming the daily steps follow a normal distribution, MLE can help estimate the true mean and standard deviation of their activity level.

In this example:
- The data consists of 10 daily step count measurements
- We assume the step counts follow a normal distribution
- MLE estimates both the mean and standard deviation
- The analysis relies solely on the observed data without prior assumptions

The MLE analysis estimates a mean daily step count of 8680 with a standard deviation of 824.62. This gives the person an understanding of their typical activity level and day-to-day variation.

![Daily Steps MLE Example](../Images/normal_mle_daily_steps.png)

### Example 5: Ball Bearing Diameter

#### Problem Statement
A quality control engineer is analyzing the diameter of manufactured ball bearings to assess production consistency. The engineer measured 10 ball bearings (in mm): 10.02, 9.98, 10.05, 9.97, 10.01, 10.03, 9.99, 10.04, 10.00, and 9.96. While the nominal diameter is 10mm, the engineer is particularly interested in estimating the variability (standard deviation) of the manufacturing process.

In this example:
- The data consists of 10 diameter measurements
- We assume the measurements follow a normal distribution
- MLE estimates the standard deviation of the manufacturing process
- The analysis relies solely on the observed data

#### Detailed Calculation

**Step 1: Gather the data**
- Observed diameters: 10.02, 9.98, 10.05, 9.97, 10.01, 10.03, 9.99, 10.04, 10.00, and 9.96 mm
- Number of observations (n) = 10

**Step 2: Calculate the sample mean**
First, we need to calculate the sample mean:

$$
\hat{\mu}_{MLE} = \frac{10.02 + 9.98 + 10.05 + 9.97 + 10.01 + 10.03 + 9.99 + 10.04 + 10.00 + 9.96}{10} = \frac{100.05}{10} = 10.005 \text{ mm}
$$

**Step 3: Calculate MLE for variance**
For a normal distribution with unknown mean, the MLE for variance is:

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2
$$

Calculating the deviations from the mean:

$$
\begin{align*}
(10.02 - 10.005)^2 &= 0.00015^2 \\
(9.98 - 10.005)^2 &= 0.00025^2 \\
(10.05 - 10.005)^2 &= 0.00045^2 \\
(9.97 - 10.005)^2 &= 0.00035^2 \\
(10.01 - 10.005)^2 &= 0.00005^2 \\
(10.03 - 10.005)^2 &= 0.00025^2 \\
(9.99 - 10.005)^2 &= 0.00015^2 \\
(10.04 - 10.005)^2 &= 0.00035^2 \\
(10.00 - 10.005)^2 &= 0.00005^2 \\
(9.96 - 10.005)^2 &= 0.00045^2
\end{align*}
$$

Sum of squared deviations = 0.00080

$$
\hat{\sigma}^2_{MLE} = \frac{0.00080}{10} = 0.000080 \text{ mm}^2
$$

**Step 4: Calculate MLE for standard deviation**

$$
\hat{\sigma}_{MLE} = \sqrt{0.000080} = 0.0283 \text{ mm}
$$

**Step 5: Interpret the results**
Based on the MLE analysis, the manufacturing process has an estimated standard deviation of 0.0283 mm. This provides a measure of the consistency of the manufacturing process, with approximately 95% of ball bearings falling within ±0.057 mm of the mean (using the 2σ rule).

![Ball Bearing Diameter MLE Example](../Images/std_mle_ball_bearing_diameter.png)

### Example 6: Battery Life

#### Problem Statement
An electronics manufacturer is testing the life of rechargeable batteries. They recorded the runtime of 12 batteries (in hours): 4.8, 5.2, 4.9, 5.1, 4.7, 5.0, 4.9, 5.3, 4.8, 5.2, 5.0, and 4.8. Assuming the battery life follows a normal distribution, MLE can help estimate the true mean and standard deviation of the battery life.

In this example:
- The data consists of 12 battery life measurements
- We assume the measurements follow a normal distribution
- MLE estimates both the mean and standard deviation
- The analysis relies solely on the observed data

The MLE analysis estimates a mean battery life of 4.98 hours with a standard deviation of 0.19 hours. This provides the manufacturer with insights about the typical battery life and its variability.

![Battery Life MLE Example](../Images/std_mle_battery_life.png)

### Example 7: Reaction Time

#### Problem Statement
A researcher is measuring reaction times in a psychology experiment. They recorded reaction times from 10 participants (in seconds): 0.32, 0.29, 0.35, 0.30, 0.28, 0.33, 0.31, 0.34, 0.30, and 0.32. Assuming the reaction times follow a normal distribution, MLE can help estimate the true mean and standard deviation of human response times.

In this example:
- The data consists of 10 reaction time measurements
- We assume the reaction times follow a normal distribution
- MLE estimates both the mean and standard deviation
- The analysis relies solely on the observed data

The MLE analysis estimates a mean reaction time of 0.31 seconds with a standard deviation of 0.02 seconds. This gives the researcher an understanding of the typical response time and its variability.

![Reaction Time MLE Example](../Images/std_mle_reaction_time.png)

### Example 8: Temperature Readings

#### Problem Statement
A climate scientist is analyzing daily temperature readings. They recorded temperatures for 12 days (in Celsius): 21.2, 20.8, 21.5, 20.9, 21.3, 21.1, 20.7, 21.0, 21.2, 20.9, 21.4, and 21.1. Assuming the temperatures follow a normal distribution, MLE can help estimate the true mean and standard deviation of the temperature readings.

In this example:
- The data consists of 12 temperature measurements
- We assume the temperatures follow a normal distribution
- MLE estimates both the mean and standard deviation
- The analysis relies solely on the observed data

The MLE analysis estimates a mean temperature of 21.1°C with a standard deviation of 0.24°C. This provides the scientist with insights about the typical temperature and its daily variation.

![Temperature Readings MLE Example](../Images/std_mle_temperature_readings.png)

## Running the Examples

You can run all the examples using the Python files:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Code/normal_mle_examples.py
python3 ML_Obsidian_Vault/Lectures/2/Code/std_mle_examples.py
```
