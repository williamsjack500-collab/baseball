**Overview**

This project was started with the intent to quantify the effectiveness of pitch sequencing during an at-bat in Major League Baseball.

**RoadMap**

First, I am building out a framework to grade batted balls. I built out an xBA model using batted balls from 2021-2026. I am currently making a v2 that accounts for ballpark dimensions, before I move on to building out a model for xSLG and xOBP. Once I have built out this framework I will transition my work in to evaluating pitch effectiveness.  

**XBA v1**  
Inputs: Launch Speed, Launch Angle, Spray Angle, Sprint Speed  
Outputs: xBA  
Log Loss: .367 (Statcast .409)  
Brier Score: .117 (Statcast .133)  
ROC AUC: .902 (Statcast .872)  

**Known Limitations**  
The model drastically underestimates deep fly balls to dead center since it is the deepest part of the field. I am currently on the process of pulling in ballpark dimensions to create a new input "Distance to fence" with the hopes that the model does a better job recognizing that batted balls with a negative value will be a hit.  

**Motivation**  
This project started because I was curious how pitch sequencing can affect an at-bat, and I wanted a way to quantify this. In the process of building this out I discovered that I first need a better framework for how I evaluate batted balls, as the current metric for xBA on Statcast does not account for spray angle. So as a first step I started building out a neural network to predict my own xBA that accounts for spray angle.  

**Research Questions**  
* What pitchers are the most/least predictable in their sequencing of pitches?
* How does changing eye level affect a batter's ability to hit the ball?
* Is a batter more likely to record a hit on a given pitch if they have already seen that pitch type in the at-bat?
